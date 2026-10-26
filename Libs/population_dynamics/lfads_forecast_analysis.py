"""Descriptive LFADS summaries and paired prior comparisons after complete replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .lfads_evaluator_compatibility import verify_legacy_evaluator
from .population_forecast_replay import UNIT_METRICS, _score_rows
from .provenance import sha256_file
from .trajectory_forecast_analysis import _write_rows, covariance_components, pair_sets

SEEDS = (41, 42, 43)
PRIOR_ARMS = ('original_prior', 'fitted_prior')
ARCHIVED_EVALUATOR = Path('Libs/population_dynamics/lfads_trajectory_evaluation.py')


def _read(path):
    return json.loads(path.read_text())


def _mean(values):
    if all(value is None for value in values):
        return None
    if any(value is None or not np.isfinite(value) for value in values):
        raise ValueError('matched finite metric availability required')
    return float(np.mean(values))


def _forecast_metrics(config):
    return ['energy_score', 'variogram_spatial', 'variogram_cross_population',
            *(f'variogram_temporal_lag_{lag}' for lag in config['temporal_lags_bins']),
            *UNIT_METRICS,
            *(f'refractory_count_violation_{w}bin' for w in config['refractory_support_windows_bins'])]


def field_mean_error_components(sampled_counts, observed_counts, unit_count, bin_seconds):
    """Partition each field's mean-rate error across time, retaining MC correction.

    The temporal centering uses the observed future only to partition error; it
    is neither a causal rate correction nor an additional forecast. Rates are
    counts per source-defined unit per second. Corrected estimates can be
    negative with a finite ensemble and are deliberately not clipped.
    """
    samples = np.asarray(sampled_counts, dtype=np.float64)
    truth = np.asarray(observed_counts, dtype=np.float64)
    units = np.asarray(unit_count)
    if (samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 1 or samples.shape[2] < 1
            or truth.shape != samples.shape[1:] or units.shape != (samples.shape[2],)
            or units.dtype.kind not in 'iu' or np.any(units <= 0)
            or not np.isfinite(bin_seconds) or bin_seconds <= 0
            or not np.isfinite(samples).all() or not np.isfinite(truth).all()
            or np.any(samples < 0) or np.any(truth < 0)):
        raise ValueError('finite nonnegative matched field counts, positive units and clock required')
    factor = units * bin_seconds
    samples, truth = samples / factor, truth / factor
    error = samples.mean(axis=0) - truth
    time_mean_error = error.mean(axis=0)
    total = np.mean(error ** 2, axis=0)
    bias = time_mean_error ** 2
    temporal = np.mean((error - time_mean_error) ** 2, axis=0)
    draws = samples.shape[0]
    total_mc = samples.var(axis=0, ddof=1).mean(axis=0) / draws
    bias_mc = samples.mean(axis=1).var(axis=0, ddof=1) / draws
    return dict(sampled_mean_rate_hz=samples.mean(axis=(0, 1)),
        observed_mean_rate_hz=truth.mean(axis=0), signed_time_mean_bias_hz=time_mean_error,
        empirical_mean_mse_hz2=total, time_mean_bias_mse_hz2=bias,
        temporal_residual_mse_hz2=temporal, ensemble_mean_mc_mse_hz2=total_mc,
        time_mean_bias_mc_mse_hz2=bias_mc,
        mean_mse_corrected_hz2=total-total_mc,
        time_mean_bias_mse_corrected_hz2=bias-bias_mc,
        temporal_residual_mse_corrected_hz2=temporal-(total_mc-bias_mc))


def _readout_inputs(evaluation, scale, checkpoint, field_view):
    # Historical analysis launchers stage this module alone. The additional
    # readout dependency is needed only for the explicit reconstruction path.
    from .lfads_readout import LFADSReadout

    checkpoint, field_view = Path(checkpoint), Path(field_view)
    for name, digest in evaluation['selected_model_artifacts'].items():
        if sha256_file(checkpoint / name) != digest:
            raise ValueError('readout checkpoint differs from paired forecasts: ' + name)
    expected = evaluation['training_binding']['episode_bindings']['train_block_014_familiar_0']
    if sha256_file(field_view / 'summary.json') != expected['field_summary_sha256']:
        raise ValueError('readout observation map summary differs from source binding')
    summary = _read(field_view / 'summary.json')
    if sha256_file(field_view / 'spatial_fields.npz') != summary['artifacts']['spatial_fields.npz']:
        raise ValueError('readout source observation map changed')
    readout = LFADSReadout.from_checkpoint(checkpoint)
    with np.load(field_view / 'spatial_fields.npz', allow_pickle=False) as data:
        units, mapping, fields = data['source_unit_ids'], data['unit_field_index'], data['field_ids']
    if (not np.array_equal(units, readout.unit_ids) or not np.array_equal(fields, scale['field_ids'])
            or mapping.shape != units.shape or mapping.dtype.kind not in 'iu'
            or np.any(mapping < 0) or np.any(mapping >= len(fields))
            or not np.array_equal(np.bincount(mapping, minlength=len(fields)), scale['unit_count'])
            or readout.bin_seconds != evaluation['training_binding']['config']['bin_ms'] / 1000):
        raise ValueError('readout unit order, source fields or clock differ')
    return readout, mapping


def observation_noise_components(samples, conditional, truth, units, bin_seconds):
    """Error of the sampled ensemble mean relative to its conditional mean."""
    factor = units * bin_seconds
    conditional_mean = conditional.mean(axis=0) / factor
    residual = (samples.mean(axis=0) - conditional.mean(axis=0)) / factor
    return dict(observation_noise_mean_mse_hz2=np.mean(residual ** 2, axis=0),
        observation_error_cross_term_hz2=2 * np.mean(residual * (conditional_mean - truth / factor), axis=0),
        expected_poisson_mean_mse_hz2=np.mean(conditional.mean(axis=0) / (len(conditional) * factor ** 2), axis=0))


def _prior_field_errors(root, evaluations, scale, readout=None, mapping=None, *, forecast_folders=None):
    """Read the already admitted paired ensembles once; retain every field."""
    fields, origins = [], []
    first = evaluations[PRIOR_ARMS[0]]
    config = first['configuration']
    bin_ms = first['training_binding']['config']['bin_ms']
    units = scale['unit_count']
    weights = units / units.sum()
    folders = forecast_folders or {arm: root / arm / 'forecasts' for arm in PRIOR_ARMS}
    for block in range(14, 18):
        for template in range(4):
            case = f'train_block_{block:03d}_familiar_{template}'
            for origin in config['origins_ms']:
                truth_reference = None
                for arm in PRIOR_ARMS:
                    name = f'{case}_origin_{origin:g}.npz'
                    with np.load(folders[arm] / name, allow_pickle=False) as data:
                        samples = data['sampled_field_counts']
                        truth = data['observed_field_counts']
                        states = data['lfads_future_generator_states'] if readout is not None else None
                    if (samples.shape[0] != config['joint_samples']
                            or samples.shape[1] * bin_ms != max(config['horizons_ms'])):
                        raise ValueError('complete registered field ensemble required')
                    if truth_reference is not None and not np.array_equal(truth, truth_reference):
                        raise ValueError('paired field observations differ')
                    truth_reference = truth
                    if readout is not None:
                        if states.shape[:2] != samples.shape[:2]:
                            raise ValueError('saved generator path differs from field samples')
                        expected, _ = readout.expected_field_counts(states, mapping, len(units))
                    for horizon in config['horizons_ms']:
                        bins = int(horizon / bin_ms)
                        if bins < 1 or bins * bin_ms != horizon:
                            raise ValueError('field error horizon must match native bin clock')
                        metrics = field_mean_error_components(samples[:, :bins], truth[:bins], units, bin_ms / 1000)
                        if readout is not None:
                            conditional = expected[:, :bins]
                            components = field_mean_error_components(conditional, truth[:bins], units, bin_ms / 1000)
                            metrics.update({'conditional_' + k: v for k, v in components.items()})
                            metrics.update(observation_noise_components(samples[:, :bins], conditional, truth[:bins], units, bin_ms / 1000))
                        identity = dict(seed=first['training_seed'], block=block, case_id=case,
                            template_id=f'familiar_{template}', origin_ms=origin, horizon_ms=horizon, arm=arm)
                        origins.append(dict(**identity, **{k: float(v @ weights) for k, v in metrics.items()}))
                        for index, field in enumerate(scale['field_ids']):
                            fields.append(dict(**identity, field_id=str(field), units=int(units[index]),
                                **{k: float(v[index]) for k, v in metrics.items()}))
    metric_names = list(metrics)
    blocks, summary = [], []
    for arm in PRIOR_ARMS:
        for horizon in config['horizons_ms']:
            for block in range(14, 18):
                members = [r for r in origins if r['arm'] == arm and r['horizon_ms'] == horizon and r['block'] == block]
                if len(members) != 12:
                    raise ValueError('complete field error block required')
                blocks.append(dict(seed=first['training_seed'], arm=arm, horizon_ms=horizon, block=block,
                    matched_episode_origins=12, **{k: _mean([r[k] for r in members]) for k in metric_names}))
    for horizon in config['horizons_ms']:
        for metric in metric_names:
            left = [r[metric] for r in blocks if r['arm'] == PRIOR_ARMS[0] and r['horizon_ms'] == horizon]
            right = [r[metric] for r in blocks if r['arm'] == PRIOR_ARMS[1] and r['horizon_ms'] == horizon]
            delta = [b-a for a,b in zip(left,right)]
            summary.append(dict(seed=first['training_seed'], horizon_ms=horizon, metric=metric,
                original_prior_mean=_mean(left), fitted_prior_mean=_mean(right),
                fitted_minus_original_mean=_mean(delta), minimum_block_difference=min(delta),
                maximum_block_difference=max(delta), independent_blocks=4, training_seeds=1))
    return {'field_error_by_field.csv': fields, 'field_error_by_origin.csv': origins,
            'field_error_by_block.csv': blocks, 'field_error_summary.csv': summary}


def _replayed_forecast(folder, replay_path, seed):
    """Reuse the existing full forecast/replay admission before aggregation."""
    evaluation = _read(folder / 'evaluation.json')
    replay = _read(replay_path)
    if (evaluation['status'] != 'EVALUATED_DEVELOPMENT' or evaluation['training_seed'] != seed
            or evaluation['model_type'] != 'lfads' or replay['status'] != 'PASS'
            or replay['training_seed'] != seed or replay['verified_origins'] != 48
            or replay['verified_score_rows'] != 144 or replay['metric_comparisons'] != 1152
            or replay['field_scores_recomputed'] != 720 or not replay['all_forecast_unit_metrics_replayed']
            or replay['evaluation_sha256'] != sha256_file(folder / 'evaluation.json')
            or replay['scores_sha256'] != sha256_file(folder / 'scores.csv')
            or evaluation['scores_sha256'] != replay['scores_sha256']):
        raise ValueError('complete source-bound LFADS forecast and independent replay required')
    if (evaluation['scientific_claim_qualified'] is not False or evaluation['formal_test_accessed'] is not False
            or evaluation['calibration_outcomes_accessed'] is not False or evaluation['evaluation_blocks'] != [14,15,16,17]):
        raise ValueError('descriptive epoch-selection scope required')
    binding = evaluation['training_binding']
    source = {key: binding[key] for key in ('acquisition_design_sha256','dataset_sha256','native_source_identity')}
    if replay['dataset_sha256'] != source['dataset_sha256']:
        raise ValueError('independent replay cohort changed')
    for name, digest in evaluation['artifacts'].items():
        if Path(name).name != name or sha256_file(folder / name) != digest:
            raise ValueError('forecast artifact changed: ' + name)
    csv_path = replay_path.with_suffix('.csv')
    if replay['csv_artifact'] != csv_path.name or replay['csv_sha256'] != sha256_file(csv_path):
        raise ValueError('independent metric comparison artifact changed')
    with np.load(folder / 'fitting_field_scale.npz', allow_pickle=False) as data:
        scale = {name:data[name] for name in ('mean_hz','standard_deviation_hz','unit_count','field_ids')}
    cases = [dict(case_id=f'train_block_{block:03d}_familiar_{template}', block=block,
                  template_id=f'familiar_{template}') for block in range(14,18) for template in range(4)]
    row_record = dict(evaluation, artifacts={k:v for k,v in evaluation['artifacts'].items() if k!='fitting_field_scale.npz'})
    indexed = _score_rows(folder, row_record, cases)
    return evaluation, replay, scale, indexed


def _graph_cost_scopes(evaluation, rows, seed):
    """Identify paid graph invocations, including inherited continuation work."""
    graphs = evaluation.get('graph_lifetimes')
    if graphs is None:
        return {}
    continuation = evaluation.get('forecast_continuation') or {}
    inherited = set(continuation.get('inherited_artifacts', {}))
    pending = set(rows)
    scopes = {}
    for index, graph in enumerate(graphs):
        origin = float(graph['origin_ms'])
        # Continuation stores inherited invocations before current invocations.
        earlier = any(float(g['origin_ms']) == origin for g in graphs[:index])
        later = any(float(g['origin_ms']) == origin for g in graphs[index + 1:])
        members = [key for key in pending if key[1] == origin]
        if earlier or later:
            if not continuation or continuation.get('previous_continuation'):
                raise ValueError('explicit single-continuation graph lineage required')
            members = [key for key in members if (rows[key]['ensemble_artifact'] in inherited) == (not earlier)]
        if (len(members) != graph['completed_cases'] or not members
                or any(float(rows[key]['graph_setup_seconds']) != graph['setup_seconds'] for key in members)):
            raise ValueError('complete measured graph invocation membership required')
        first = min(members)
        for key in members:
            scopes[key] = dict(graph_setup_id=f'seed{seed}:graph{index}',
                graph_setup_cases=len(members),
                graph_setup_source='parent' if rows[key]['ensemble_artifact'] in inherited else 'current',
                graph_setup_charged_seconds=graph['setup_seconds'] if key == first else 0.)
        pending.difference_update(members)
    if pending:
        raise ValueError('measured graph lifetime missing origins')
    return scopes


def analyze(root, output, *, forecast_inputs=None, method='lfads'):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if forecast_inputs is not None and set(forecast_inputs) != set(SEEDS):
        raise ValueError('exact three-seed explicit forecast inputs required')
    if not method or not isinstance(method, str):
        raise ValueError('explicit baseline method label required')
    reports, bindings = {}, {}
    shared_config = shared_source = shared_scale = None
    metrics = None
    all_rows, block_rows, summary, covariance, cov_arrays, costs = [], [], [], [], {}, []
    for seed in SEEDS:
        folder, replay_path = ((root / f'seed_{seed}_forecast', root / f'seed_{seed}_replay.json')
                               if forecast_inputs is None else map(Path, forecast_inputs[seed]))
        evaluation, replay, scale, indexed = _replayed_forecast(folder, replay_path, seed)
        origin_rows = {(row['case_id'], float(row['origin_ms'])): row for row in indexed.values()
                       if float(row['horizon_ms']) == max(evaluation['configuration']['horizons_ms'])}
        scopes = _graph_cost_scopes(evaluation, origin_rows, seed)
        binding = evaluation['training_binding']
        source = {key: binding[key] for key in ('acquisition_design_sha256','dataset_sha256','native_source_identity')}
        config = evaluation['configuration']
        if shared_config is None:
            shared_config, shared_source = config, source
        if shared_config != config or shared_source != source:
            raise ValueError('matched source and forecast protocol across training seeds required')
        if shared_scale is None:
            shared_scale = scale
        if any(not np.array_equal(scale[k], shared_scale[k]) for k in scale):
            raise ValueError('fitting scale or observation map differs across seeds')
        metrics = _forecast_metrics(config)
        rows, grouped, ensembles = [], defaultdict(list), {}
        for raw in indexed.values():
            row = {key:raw[key] for key in ('case_id','template_id','method','ensemble_artifact')}
            row['method'] = method
            row.update(seed=int(raw['seed']), block=int(raw['block']), origin_ms=float(raw['origin_ms']), horizon_ms=float(raw['horizon_ms']))
            for key in metrics:
                row[key] = None if raw[key]=='' else float(raw[key])
                if row[key] is not None and not np.isfinite(row[key]):
                    raise ValueError('nonfinite replayed forecast metric')
            name = row['ensemble_artifact']
            if name not in ensembles:
                with np.load(folder/name, allow_pickle=False) as data:
                    factor = scale['unit_count'] * binding['config']['bin_ms'] / 1000
                    ensembles[name] = ((data['sampled_field_counts']/factor-scale['mean_hz'])/scale['standard_deviation_hz'],
                                       (data['observed_field_counts']/factor-scale['mean_hz'])/scale['standard_deviation_hz'])
                costs.append(dict(seed=seed, case_id=row['case_id'], origin_ms=row['origin_ms'],
                    **{k:None if v=='' else float(v) for k,v in raw.items() if k.endswith('_seconds') or k.endswith('_bytes')},
                    **scopes.get((row['case_id'], row['origin_ms']), {})))
            grouped[(row['template_id'],row['origin_ms'],row['horizon_ms'])].append(row)
            rows.append(row)
        all_rows.extend(rows)
        for block in range(14,18):
            for horizon in config['horizons_ms']:
                members=[r for r in rows if r['block']==block and r['horizon_ms']==horizon]
                if len(members)!=12:raise ValueError('complete four-template/three-origin block required')
                block_rows.append(dict(seed=seed, block=block, method=method,horizon_ms=horizon,
                                       **{k:_mean([r[k] for r in members]) for k in metrics}))
        pairs=pair_sets(scale['field_ids'])
        for (template,origin,ms),members in sorted(grouped.items()):
            members.sort(key=lambda r:r['block'])
            if [r['block'] for r in members]!=[14,15,16,17]:raise ValueError('four independent selection blocks required')
            horizon=int(ms/binding['config']['bin_ms'])
            sample=np.stack([ensembles[r['ensemble_artifact']][0][:,:horizon] for r in members])
            truth=np.stack([ensembles[r['ensemble_artifact']][1][:horizon] for r in members])
            requests=[(name,ij[:,0],ij[:,1],0) for name,ij in pairs.items()]
            requests += [(f'temporal_lag_{lag}',np.arange(sample.shape[-1]),np.arange(sample.shape[-1]),lag)
                         for lag in config['temporal_lags_bins'] if lag<horizon]
            for name,left,right,lag in requests:
                observed,predicted,within,between=covariance_components(sample,truth,left,right,lag)
                payload=np.stack((observed,predicted,within,between))
                if not np.isfinite(payload).all():raise ValueError('nonfinite covariance diagnostic')
                key=f'g{len(covariance)}';cov_arrays[key]=payload
                covariance.append(dict(seed=seed,method=method,template_id=template,origin_ms=origin,horizon_ms=ms,
                    statistic=name,independent_blocks=4,pairs=len(left),phases=horizon-lag,array_key=key,
                    observed_covariance_mean=float(observed.mean()),predicted_covariance_mean=float(predicted.mean()),
                    covariance_mse=float(((predicted-observed)**2).mean()),covariance_mae=float(np.abs(predicted-observed).mean())))
        bindings[str(seed)]=dict(evaluation_sha256=sha256_file(folder/'evaluation.json'),replay_sha256=sha256_file(replay_path),
            selected_model_artifacts=evaluation['selected_model_artifacts'], completed_training_epochs=evaluation['completed_training_epochs'],
            selected_epoch=evaluation['selected_epoch'], training_stop_reason=evaluation['training_stop_reason'],
            training_config=binding['config'], artifacts=evaluation['artifacts'])
        bindings[str(seed)].update({key:evaluation[key] for key in (
            'conditional_prior', 'prior_refit', 'effective_training_epoch_budget', 'training_continuation',
            'forecast_continuation', 'graph_lifetimes') if key in evaluation})
    for horizon in shared_config['horizons_ms']:
        for metric in metrics:
            values=[_mean([r[metric] for r in block_rows if r['seed']==seed and r['horizon_ms']==horizon]) for seed in SEEDS]
            value=_mean(values)
            if value is not None:
                summary.append(dict(method=method,horizon_ms=horizon,metric=metric,mean=value,
                    minimum_seed_mean=min(values),maximum_seed_mean=max(values),independent_blocks=4,training_seeds=3,
                    interpretation='development_descriptive'))
    cov_groups=defaultdict(list)
    for row in covariance:cov_groups[(row['horizon_ms'],row['statistic'])].append(row)
    cov_summary=[]
    for (horizon,statistic),members in sorted(cov_groups.items()):
        cov_summary.append(dict(method=method,horizon_ms=horizon,statistic=statistic,training_seeds=3,
            independent_blocks=4,matched_template_origins=12,**{k:_mean([r[k] for r in members]) for k in
            ('observed_covariance_mean','predicted_covariance_mean','covariance_mse','covariance_mae')}))
    output.mkdir(parents=True,exist_ok=False)
    tables={'forecast_all_seeds.csv':all_rows,'forecast_by_block.csv':block_rows,'forecast_summary.csv':summary,
            'covariance_diagnostic.csv':covariance,'covariance_summary.csv':cov_summary,'costs_by_origin.csv':costs}
    for name,rows in tables.items():_write_rows(output/name,rows)
    np.savez_compressed(output/'covariance_arrays.npz',**cov_arrays)
    report=dict(status='DESCRIBED_REPLAYED_DEVELOPMENT',training_seeds=list(SEEDS),model=method,independent_blocks=4,
        connectivity_instances=1,score_rows=len(all_rows),forecast_origins=len(costs),configuration=shared_config,
        source_binding=shared_source,model_seed_bindings=bindings,all_unit_metrics_independently_replayed=True,
        covariance_scope='total predictive covariance across four initial/background selection blocks at fixed template/origin/phase',
        covariance_array_axis=['observed','predicted_total','mean_within','between_sample_means'],
        uncertainty='descriptive epoch-selection blocks; no inferential CI/p/q or identified conditional covariance claim',
        scientific_claim_qualified=False,convergence_established=False,formal_test_accessed=False,
        costs='one row per500ms origin; raw graph setup retained per case; graph_setup_charged_seconds counts each identified invocation once, including inherited work; TensorFlow GPU allocation peak unmeasured; fitting and failed-attempt costs remain separate; no end-to-end cost claim',
        implementation_sha256=sha256_file(Path(__file__)),artifacts={n:sha256_file(output/n) for n in [*tables,'covariance_arrays.npz']})
    (output/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _comparison_completion(root, seed, prior, accepted_files):
    """Bind existing launcher completion receipts when supplied with the arms."""
    receipts = {}
    for name in ('execution.json', 'delivery.json'):
        path = root / name
        if not path.exists():
            continue
        record = _read(path)
        arms = record.get('comparison_arms', [])
        cache = record.get('comparison_cache', {})
        if (record.get('status') != 'COMPARED_PRIORS_DEVELOPMENT'
                or record.get('training_seed') != seed
                or [row.get('arm') for row in arms] != list(PRIOR_ARMS)
                or any(row.get('status') != 'VERIFIED'
                       or row.get('forecast_status') != 'EVALUATED_DEVELOPMENT'
                       or row.get('replay_status') != 'PASS'
                       or row.get('forecasts') != row['arm'] + '/forecasts'
                       or row.get('replay') != row['arm'] + '/replay.json' for row in arms)
                or cache.get('status') != 'VERIFIED'
                or cache.get('verified_posterior_episodes') != 56
                or cache.get('prior_binding') != prior):
            raise ValueError('completed paired prior execution contract required')
        if name == 'delivery.json':
            files = record['files']
            entries = {row['path']: row for row in files}
            if len(entries) != len(files):
                raise ValueError('duplicate prior comparison delivery entries')
            for relative, digest in accepted_files.items():
                entry = entries.get(relative, {})
                if (entry.get('sha256') != digest
                        or entry.get('bytes') != (root / relative).stat().st_size):
                    raise ValueError('prior comparison delivery binding changed: ' + relative)
        receipts[name] = sha256_file(path)
        accepted_files[name] = receipts[name]
    return receipts


def _forecast_files(folder, replay_path, evaluation, replay):
    files = {folder / name: digest for name, digest in evaluation['artifacts'].items()}
    files.update({folder / 'evaluation.json': replay['evaluation_sha256'],
                  folder / 'scores.csv': evaluation['scores_sha256'],
                  replay_path: sha256_file(replay_path), replay_path.with_suffix('.csv'): replay['csv_sha256']})
    return files


def _delivered_files(root, record, accepted_files):
    entries = {}
    for row in record['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in entries:
            raise ValueError('unique relative prior delivery paths required')
        entries[str(relative)] = row
    for path, digest in accepted_files.items():
        name = str(path.relative_to(root))
        row = entries.get(name, {})
        if row.get('sha256') != digest or row.get('bytes') != path.stat().st_size:
            raise ValueError('delivered prior input changed: ' + name)
    return entries


def _original_completion(folder, replay_path, evaluation, replay, accepted_files):
    """Bind a reused original arm to its existing archive, including any rescore."""
    archive = next((parent for parent in (folder, *folder.parents)
                    if replay_path.is_relative_to(parent) and (parent / 'delivery.json').is_file()), None)
    if archive is None:
        raise ValueError('reused original forecast and replay require their archive delivery')
    record = _read(archive / 'delivery.json')
    # The historical budget archive has no top-level exit_code. Its successful
    # archive status and the independently checked replay establish completion.
    if (record.get('status') not in ('ARCHIVED_REPLAYED_BUDGET_COMPARISON', 'COMPARED_PRIORS_DEVELOPMENT')
            or ('exit_code' in record and record['exit_code'] != 0)):
        raise ValueError('completed original-prior archive required')
    _delivered_files(archive, record, accepted_files)
    rescore = evaluation.get('numerical_rescore')
    archived_evaluator_proof = None
    if rescore is not None or replay.get('numerical_rescore') is not None:
        if (not isinstance(rescore, dict) or replay.get('numerical_rescore') != rescore
                or rescore.get('schema') != 'lfads_exact_count_correction_rescore_v1'
                or rescore.get('unchanged_estimand') is not True
                or rescore.get('unchanged_output_dtype') != evaluation['configuration']['unit_metric_dtype']
                or rescore.get('unchanged_metric_tolerance') != replay.get('metric_tolerance')):
            raise ValueError('reused original rescore and independent replay must preserve the same protocol')
        if 'verified_legacy_evaluator' in rescore:
            proof = rescore['verified_legacy_evaluator']
            if not isinstance(proof, dict) or proof.get('numerical_program_ast_equal') is not True:
                raise ValueError('reused original rescore requires a valid evaluator proof')
        else:
            # The original budget evaluator needed no bridge to itself. Verify
            # its delivered source against today's default-prior program without
            # rewriting the historical rescore or substituting arithmetic proof.
            if any(key in document for document in (evaluation, replay)
                   for key in ('prior_refit', 'conditional_prior', 'forecast_continuation')):
                raise ValueError('archived evaluator proof requires the original default forecast path')
            source = archive / ARCHIVED_EVALUATOR
            if not source.is_file():
                raise ValueError('delivered original evaluator source required')
            digest = evaluation.get('evaluator_implementation_sha256')
            _delivered_files(archive, record, {source: digest})
            archived_evaluator_proof = verify_legacy_evaluator(
                digest, source, Path(__file__).with_name('lfads_trajectory_evaluation.py'))
            accepted_files[source] = digest
    return dict(archive=str(archive), delivery_sha256=sha256_file(archive / 'delivery.json'),
                forecast=str(folder), replay=str(replay_path), numerical_rescore=rescore,
                verified_archived_evaluator=archived_evaluator_proof)


def _stationary_completion(root, record, evaluation, prior, accepted_files, fit_run):
    """Reuse the completed native intake; do not regenerate posterior paths."""
    seed = evaluation['training_seed']
    expected = dict(status='REPLAYED_STATIONARY_PRIOR_CANDIDATE', exit_code=0,
        training_seed=seed, prior_variant='stationary', completed_origins=48,
        original_reference_reexecuted=False, neural_training_updates=0,
        scientific_claim_qualified=False, formal_test_accessed=False, mean_run=None)
    execution_path = root / 'execution.json'
    execution = _read(execution_path)
    for document in (record, execution):
        if any(document.get(key) != value for key, value in expected.items()):
            raise ValueError('completed fitted-only forecast and replay required')
    intake = record.get('stationary_prior_intake', {})
    if (intake.get('status') != 'VERIFIED' or intake.get('complete_episodes') != 56
            or type(intake.get('frozen_global_variables_verified')) is not int
            or intake['frozen_global_variables_verified'] <= 0
            or type(intake.get('training_updates_before_and_after')) is not int
            or intake['training_updates_before_and_after'] < 0
            or any(execution.get(key) != record.get(key) for key in
                   ('reference_run', 'reference_delivery_sha256', 'network_budget_epochs', 'stationary_prior_intake'))):
        raise ValueError('complete frozen stationary-prior intake proof required')
    accepted_files = dict(accepted_files)
    accepted_files[execution_path] = sha256_file(execution_path)
    _delivered_files(root, record, accepted_files)

    parent = Path(fit_run) if fit_run is not None else Path(record['reference_run'])
    parent = parent.resolve()
    parent_path = parent / 'delivery.json'
    if sha256_file(parent_path) != record['reference_delivery_sha256']:
        raise ValueError('fitted-only candidate parent fit delivery changed')
    fit = _read(parent_path)
    commands = [row for row in fit.get('commands', []) if row.get('label') == 'posterior_fit']
    budget = {'budget_20': 20, 'budget_80': 80}.get(fit.get('network_variant'))
    if (fit.get('status') != 'FITTED_PRIOR_ONLY' or fit.get('action') != 'fit'
            or fit.get('training_seed') != seed or fit.get('staged_inputs', {}).get('status') != 'VERIFIED'
            or budget is None or record.get('network_budget_epochs') != budget
            or evaluation.get('effective_training_epoch_budget', budget) != budget
            or len(commands) != 1 or commands[0].get('status') != 'PASS' or commands[0].get('exit_code') != 0):
        raise ValueError('completed matching parent stationary fit required')
    ledger_path, parameters_path = parent / 'cache/posterior_cache.json', parent / 'cache/prior_parameters.npz'
    ledger_digest, parameters_digest = sha256_file(ledger_path), sha256_file(parameters_path)
    entries = _delivered_files(parent, fit, {ledger_path: ledger_digest, parameters_path: parameters_digest})
    ledger = _read(ledger_path)
    binding = ledger['binding']
    cases = [f'train_block_{block:03d}_familiar_{template}' for block in range(14) for template in range(4)]
    cohort_keys = ('acquisition_design_sha256', 'acquisition_ledger_sha256', 'dataset_sha256',
                   'episode_table_sha256', 'native_source_identity', 'episode_bindings')
    expected_binding = dict(training_seed=seed, training_binding=evaluation['training_binding'],
        training_run_sha256=evaluation['training_run_sha256'], selected_model_artifacts=evaluation['selected_model_artifacts'],
        selected_epoch=prior['selected_epoch'], inference_migration=evaluation['inference_migration'],
        cohort_binding={key: evaluation['training_binding'][key] for key in cohort_keys},
        case_ids=cases, fitting_blocks=list(range(14)), excluded_selection_blocks=list(range(14, 18)),
        joint_samples=16, complete_bins=1200,
        selected_training_updates=intake['training_updates_before_and_after'])
    proof_keys = ('frozen_global_variables_verified', 'training_updates_before_and_after')
    if (ledger.get('status') != 'FITTED_FROZEN_NETWORK_AR_PRIOR'
            or ledger.get('fitting_cohort_extracted') is not True or ledger.get('model_parameters_updated') is not False
            or ledger.get('formal_test_accessed') is not False or ledger.get('complete_episodes') != 56
            or ledger.get('posterior_paths') != 896
            or ledger_digest != prior.get('posterior_cache_sha256')
            or parameters_digest != prior.get('prior_parameters_sha256')
            or ledger.get('prior_parameters_sha256') != parameters_digest
            or any(binding.get(key) != value for key, value in expected_binding.items())
            or any(ledger.get(key) != intake[key] for key in proof_keys)
            or [row.get('case_id') for row in ledger.get('episodes', [])] != cases):
        raise ValueError('fitted-only candidate cache, checkpoint, or cohort binding changed')
    # The forecast intake already checked these arrays natively. Bind its exact
    # immutable ledger and delivery entries rather than loading unused arrays.
    for case, row in zip(cases, ledger['episodes']):
        name = case + '.npz'
        declared = entries.get('cache/' + name, {})
        if (row.get('path') != name or row.get('training_seed') != seed
                or not isinstance(declared.get('sha256'), str) or len(declared['sha256']) != 64
                or row.get('sha256') != declared['sha256']
                or any(row.get(key) != intake[key] for key in proof_keys)):
            raise ValueError('fitted-only candidate episode proof changed')
    return dict(delivery_sha256=sha256_file(root / 'delivery.json'), execution_sha256=sha256_file(execution_path),
                fit_run=str(parent), fit_delivery_sha256=sha256_file(parent_path),
                posterior_cache_sha256=ledger_digest, prior_parameters_sha256=parameters_digest,
                completed_native_intake_reused=True)


def _verify_prior_graph_state(evaluation):
    """Check every graph segment and complete case coverage at each logical origin."""
    origins = evaluation['configuration']['origins_ms']
    graphs = evaluation.get('graph_lifetimes', [])
    continuation = evaluation.get('forecast_continuation')
    if continuation is not None:
        inherited = continuation.get('inherited_origins') if isinstance(continuation, dict) else None
        if (type(inherited) is not int or not 1 <= inherited <= evaluation['completed_origins']
                or continuation.get('inherited_score_rows') != inherited * len(evaluation['configuration']['horizons_ms'])
                or continuation.get('numerical_program_without_continuation_equal') is not True
                or continuation.get('full_48_origin_replay_required') is not True
                or continuation.get('current_evaluator_sha256') != evaluation['evaluator_implementation_sha256']):
            raise ValueError('verified forecast continuation required for split prior graph lifetimes')
    elif len(graphs) != len(origins):
        raise ValueError('fitted prior must preserve nonprior network state at all origins')
    counts = dict.fromkeys(origins, 0)
    ranks = {origin: index for index, origin in enumerate(origins)}
    order = []
    for graph in graphs:
        origin, completed = graph.get('origin_ms'), graph.get('completed_cases')
        state = graph.get('prior_refit_state_check', {})
        if (origin not in counts or type(completed) is not int or not 0 <= completed <= 16
                or state.get('assigned_scalars') != 8
                or state.get('nonprior_weights_and_optimizer_state_exact') is not True):
            raise ValueError('fitted prior must preserve nonprior network state at all origins')
        counts[origin] += completed
        order.append(ranks[origin])
    # A stopped graph with zero committed cases still represents real setup
    # cost. Validate it and retain it; only case coverage is summed here.
    if order != sorted(order) or any(count != 16 for count in counts.values()):
        raise ValueError('fitted prior must preserve nonprior network state at all origins')


def compare_priors(root, output, *, decompose_field_errors=False, readout_checkpoint=None, field_view=None,
                   original_forecast=None, original_replay=None, fit_run=None):
    """Pair one seed's two completed prior arms on the four selection blocks."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (readout_checkpoint is None) != (field_view is None):
        raise ValueError('readout checkpoint and bound source field view are both required')
    if (original_forecast is None) != (original_replay is None):
        raise ValueError('reused original forecast and replay paths must be supplied together')
    reuse = original_forecast is not None
    if fit_run is not None and not reuse:
        raise ValueError('parent fit override requires explicit original forecast and replay')
    locations = {arm: (root / arm / 'forecasts', root / arm / 'replay.json') for arm in PRIOR_ARMS}
    candidate_record = None
    if reuse:
        root = root.resolve()
        candidate_record = _read(root / 'delivery.json')
        standalone = candidate_record.get('status') == 'REPLAYED_STATIONARY_PRIOR_CANDIDATE'
        if not standalone and candidate_record.get('status') != 'COMPARED_PRIORS_DEVELOPMENT':
            raise ValueError('completed fitted candidate archive required')
        if fit_run is not None and not standalone:
            raise ValueError('parent fit override is only for a fitted-only candidate')
        prefix = 'candidate' if standalone else 'fitted_prior'
        locations = {'original_prior': (Path(original_forecast).resolve(), Path(original_replay).resolve()),
                     'fitted_prior': (root / prefix / 'forecasts', root / prefix / 'replay.json')}
    seed = _read(locations['original_prior'][0] / 'evaluation.json')['training_seed']
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('one registered training seed required for prior comparison')
    evaluations, replays, arm_rows, bindings, accepted_files, arm_files = {}, {}, {}, {}, {}, {}
    shared_scale = None
    for arm in PRIOR_ARMS:
        folder, replay_path = locations[arm]
        evaluation, replay, scale, indexed = _replayed_forecast(folder, replay_path, seed)
        if (evaluation.get('completed_origins') != 48 or evaluation.get('score_rows') != 144
                or evaluation.get('episodes') != 16 or evaluation.get('independent_blocks') != 4):
            raise ValueError('complete 48-origin 16-episode prior arm required')
        if shared_scale is None:
            shared_scale = scale
        elif any(not np.array_equal(scale[key], shared_scale[key]) for key in scale):
            raise ValueError('prior arms have different fitting scale or observation map')
        config = evaluation['configuration']
        metrics = _forecast_metrics(config)
        rows = {}
        for raw in indexed.values():
            key = (int(raw['seed']), int(raw['block']), raw['case_id'], raw['template_id'],
                   float(raw['origin_ms']), float(raw['horizon_ms']))
            if key in rows or raw['method'] != 'lfads':
                raise ValueError('duplicate or non-LFADS prior comparison row')
            values = {metric: None if raw[metric] == '' else float(raw[metric]) for metric in metrics}
            if any(value is not None and not np.isfinite(value) for value in values.values()):
                raise ValueError('nonfinite paired prior metric')
            rows[key] = values
        expected = {(seed, block, f'train_block_{block:03d}_familiar_{template}', f'familiar_{template}',
                     float(origin), float(horizon))
                    for block in range(14, 18) for template in range(4)
                    for origin in config['origins_ms'] for horizon in config['horizons_ms']}
        if len(expected) != 144 or set(rows) != expected:
            raise ValueError('complete matched prior comparison keys required')
        evaluations[arm], arm_rows[arm] = evaluation, rows
        replays[arm] = replay
        replay_digest = sha256_file(replay_path)
        bindings[arm] = dict(evaluation_sha256=replay['evaluation_sha256'],
            replay_sha256=replay_digest, scores_sha256=evaluation['scores_sha256'],
            selected_model_artifacts=evaluation['selected_model_artifacts'], prior_refit=evaluation.get('prior_refit'))
        accepted_files.update({arm + '/forecasts/' + name: digest for name, digest in evaluation['artifacts'].items()})
        accepted_files.update({arm + '/forecasts/evaluation.json': replay['evaluation_sha256'],
            arm + '/forecasts/scores.csv': evaluation['scores_sha256'], arm + '/replay.json': replay_digest,
            arm + '/replay.csv': replay['csv_sha256']})
        arm_files[arm] = _forecast_files(folder, replay_path, evaluation, replay)
        if evaluation.get('prior_refit') != replay.get('prior_refit'):
            raise ValueError('forecast and replay prior identity differ')
    original, fitted = (evaluations[arm] for arm in PRIOR_ARMS)
    matched_fields = ('training_binding', 'training_run_sha256', 'selected_checkpoint',
                      'selected_training_sha256', 'selected_model_artifacts', 'selected_epoch',
                      'native_source_identity', 'configuration', 'inference_migration')
    if any(original[key] != fitted[key] for key in matched_fields):
        raise ValueError('prior arms must share checkpoint, source, and forecast protocol')
    prior = fitted.get('prior_refit')
    if ('prior_refit' in original or not isinstance(prior, dict)
            or prior.get('fitted_parameters') != 8 or prior.get('fitting_episodes') != 56
            or prior.get('training_seed') != seed):
        raise ValueError('original and eight-scalar fitting-only prior arms required')
    _verify_prior_graph_state(fitted)
    if set(arm_rows['original_prior']) != set(arm_rows['fitted_prior']):
        raise ValueError('unpaired prior comparison rows')
    for key, values in arm_rows['original_prior'].items():
        for metric, value in values.items():
            if (value is None) != (arm_rows['fitted_prior'][key][metric] is None):
                raise ValueError('paired prior metric availability differs')
    if reuse:
        if 'conditional_prior' in original or 'conditional_prior' in fitted:
            raise ValueError('reused original and stationary fitted prior must not include a conditional-mean treatment')
        for key in ('metric_tolerance', 'field_tolerance'):
            if (not isinstance(replays['original_prior'].get(key), dict)
                    or replays['original_prior'][key] != replays['fitted_prior'].get(key)):
                raise ValueError('reused prior arms have different replay tolerances')
        original_proof = _original_completion(*locations['original_prior'], original,
            replays['original_prior'], arm_files['original_prior'])
        if standalone:
            candidate_proof = _stationary_completion(root, candidate_record, fitted, prior,
                                                     arm_files['fitted_prior'], fit_run)
        else:
            used = {str(path.relative_to(root)): digest for path, digest in arm_files['fitted_prior'].items()}
            candidate_proof = _comparison_completion(root, seed, prior, used)
        completion = dict(original=original_proof, fitted=candidate_proof)
    else:
        completion = _comparison_completion(root, seed, prior, accepted_files)
    readout = mapping = None
    if readout_checkpoint is not None:
        readout, mapping = _readout_inputs(original, shared_scale, readout_checkpoint, field_view)
        decompose_field_errors = True
    blocks, differences, summary = [], [], []
    for block in range(14, 18):
        for horizon in config['horizons_ms']:
            paired = {}
            for arm in PRIOR_ARMS:
                members = [row for key, row in arm_rows[arm].items() if key[1] == block and key[5] == horizon]
                if len(members) != 12:
                    raise ValueError('complete four-template three-origin block required')
                paired[arm] = {metric: _mean([row[metric] for row in members]) for metric in metrics}
                blocks.append(dict(seed=seed, block=block, arm=arm, method='lfads', horizon_ms=horizon,
                    matched_episode_origins=12, **paired[arm]))
            for metric in metrics:
                left, right = (paired[arm][metric] for arm in PRIOR_ARMS)
                delta = None if left is None else right - left
                if delta is not None and not np.isfinite(delta):
                    raise ValueError('nonfinite aggregated prior difference')
                differences.append(dict(seed=seed, block=block, horizon_ms=horizon, metric=metric,
                    original_prior=left, fitted_prior=right, fitted_minus_original=delta,
                    available=delta is not None, matched_episode_origins=12, training_seeds=1))
    for horizon in config['horizons_ms']:
        for metric in metrics:
            members = [row for row in differences if row['horizon_ms'] == horizon and row['metric'] == metric]
            deltas = [row['fitted_minus_original'] for row in members]
            summary.append(dict(seed=seed, method='lfads', horizon_ms=horizon, metric=metric,
                original_prior_mean=_mean([row['original_prior'] for row in members]),
                fitted_prior_mean=_mean([row['fitted_prior'] for row in members]),
                fitted_minus_original_mean=_mean(deltas),
                minimum_block_difference=None if deltas[0] is None else min(deltas),
                maximum_block_difference=None if deltas[0] is None else max(deltas),
                available=deltas[0] is not None, independent_blocks=4, training_seeds=1,
                interpretation='development_descriptive'))
    output.mkdir(parents=True, exist_ok=False)
    tables = {'forecast_by_block.csv': blocks, 'forecast_paired_block_differences.csv': differences,
              'forecast_summary.csv': summary}
    if decompose_field_errors:
        tables.update(_prior_field_errors(root, evaluations, shared_scale, readout, mapping,
                                         forecast_folders={arm: locations[arm][0] for arm in PRIOR_ARMS}))
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    report = dict(status='DESCRIBED_REPLAYED_PRIOR_COMPARISON', model='lfads', training_seed=seed,
        training_seeds=1, independent_blocks=4, connectivity_instances=1, episodes_per_arm=16,
        forecast_origins_per_arm=48, score_rows_per_arm=144, matched_score_rows=144,
        configuration=config, metrics=metrics, training_binding=original['training_binding'],
        arm_bindings=bindings, completion_contracts=completion,
        pairing_key=['seed', 'block', 'case_id', 'template_id', 'origin_ms', 'horizon_ms'],
        aggregation='equal mean of four templates times three origins within each block and horizon; equal mean of four paired block differences',
        difference='fitted_prior minus original_prior',
        unavailable_metrics='matched empty metrics retained as unavailable; no imputation or subset selection',
        uncertainty='single-seed descriptive epoch-selection comparison; no inferential CI/p/q',
        all_unit_metrics_independently_replayed=True, scientific_claim_qualified=False,
        convergence_established=False, formal_test_accessed=False, calibration_outcomes_accessed=False,
        implementation_sha256=sha256_file(Path(__file__)),
        artifacts={name: sha256_file(output / name) for name in tables})
    if reuse:
        report.update(original_prior_reused=True, new_forecast_origins=0,
                      original_forecast=str(locations['original_prior'][0]),
                      original_replay=str(locations['original_prior'][1]))
    if decompose_field_errors:
        report['field_error_decomposition'] = dict(
            observations='source-defined field counts per member unit per second; not single-unit MSE',
            averaging='unit-count-weighted fields; equal time bins, origins, templates and four blocks',
            identity='mean_mse = squared time-mean error + time-centered residual mse',
            finite_ensemble_correction='sample variance across draws / draws; time-mean correction includes temporal covariance',
            negative_corrected_estimates_retained=True,
            uses_future_truth_for_error_partition_only=True,
            rate_calibration_fitted=False, predictive_intensities_available=False,
            latent_dynamics_vs_observation_process_identified=False,
            new_forecast_origins=0, field_rows=len(tables['field_error_by_field.csv']),
            origin_horizon_arm_rows=len(tables['field_error_by_origin.csv']))
    if readout is not None:
        report['conditional_mean_reconstruction'] = dict(
            source='saved future generator states and unchanged selected checkpoint readout',
            units='expected counts per native bin before source field projection',
            generator_projection_normalization='columns; squared norm floor1e-12 before square root',
            source_log_rate_guard='reject nonfinite or abs(log_count)>=source clip value',
            observation_error_identity='sample mean error squared = conditional mean error squared + sampling residual squared + cross term',
            expected_sampling_noise='Poisson field sum variance divided by draws and squared field size and bin seconds',
            posterior_or_dynamics_rerun=False, new_spike_samples=0, rate_calibration_fitted=False,
            decoder_vs_latent_dynamics_identified=False,
            checkpoint_artifacts=original['selected_model_artifacts'],
            field_view_sha256=sha256_file(Path(field_view) / 'summary.json'),
            implementation_sha256=sha256_file(Path(__file__).with_name('lfads_readout.py')))
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def _matrix_cell(archive, source_delivery=None):
    """Read only authenticated, completed comparison tables and their receipts."""
    delivery = _read(archive / 'delivery.json')
    if (delivery.get('status') != 'DESCRIBED_REPLAYED_PRIOR_COMPARISON'
            or delivery.get('exit_code') != 0 or delivery.get('analysis_exit_code') != 0):
        raise ValueError('completed prior analysis delivery required')
    report_path = archive / 'analysis/analysis.json'
    table_path = archive / 'analysis/forecast_paired_block_differences.csv'
    consumed = {path: sha256_file(path) for path in (report_path, table_path)}
    _delivered_files(archive, delivery, consumed)
    report = _read(report_path)
    expected = dict(status='DESCRIBED_REPLAYED_PRIOR_COMPARISON', model='lfads',
        training_seeds=1, independent_blocks=4, connectivity_instances=1, episodes_per_arm=16,
        forecast_origins_per_arm=48, score_rows_per_arm=144, matched_score_rows=144,
        all_unit_metrics_independently_replayed=True, scientific_claim_qualified=False,
        convergence_established=False, formal_test_accessed=False, calibration_outcomes_accessed=False,
        original_prior_reused=True, new_forecast_origins=0, difference='fitted_prior minus original_prior')
    seed = report.get('training_seed')
    if (any(report.get(key) != value for key, value in expected.items())
            or type(seed) is not int or seed not in SEEDS or delivery.get('training_seed') != seed
            or report.get('training_binding', {}).get('seed') != seed
            or report.get('artifacts', {}).get(table_path.name) != consumed[table_path]):
        raise ValueError('complete descriptive comparison report and table binding required')
    source_path = (Path(source_delivery) if source_delivery is not None
                   else Path(delivery['source_comparison']) / 'delivery.json')
    source_digest = sha256_file(source_path)
    source = _read(source_path)
    fitted = report.get('completion_contracts', {}).get('fitted', {})
    if (source_digest != delivery.get('source_delivery_sha256')
            or source_digest != fitted.get('delivery_sha256', fitted.get('delivery.json'))
            or source.get('training_seed') != seed):
        raise ValueError('analysis source completion binding changed')
    if source.get('status') == 'COMPARED_PRIORS_DEVELOPMENT':
        budget = {'budget_20': 20, 'budget_80': 80}.get(source.get('network_variant'))
    elif source.get('status') == 'REPLAYED_STATIONARY_PRIOR_CANDIDATE' and source.get('exit_code') == 0:
        budget = source.get('network_budget_epochs')
    else:
        raise ValueError('completed bound prior-comparison source required')
    if type(budget) is not int or budget not in (20, 80):
        raise ValueError('bound source must identify the 20 or 80 epoch network budget')
    config, metrics = report['configuration'], report['metrics']
    if metrics != _forecast_metrics(config) or len(set(metrics)) != len(metrics):
        raise ValueError('complete registered forecast metrics required')
    expected_keys = {(block, float(horizon), metric) for block in range(14, 18)
                     for horizon in config['horizons_ms'] for metric in metrics}
    rows = {}
    with table_path.open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row['block']), float(row['horizon_ms']), row['metric'])
            values = [None if row[name] == '' else float(row[name]) for name in
                      ('original_prior', 'fitted_prior', 'fitted_minus_original')]
            if (key in rows or int(row['seed']) != seed or row['training_seeds'] != '1'
                    or row['matched_episode_origins'] != '12' or row['available'] not in ('True', 'False')
                    or any(value is not None and not np.isfinite(value) for value in values)
                    or len({value is None for value in values}) != 1
                    or (row['available'] == 'True') != (values[0] is not None)
                    or (values[0] is not None and values[2] != values[1] - values[0])):
                raise ValueError('complete finite paired block values and matched availability required')
            rows[key] = dict(network_budget_epochs=budget, seed=seed, block=key[0], horizon_ms=key[1],
                metric=key[2], original_prior=values[0], fitted_prior=values[1], fitted_minus_original=values[2],
                available=values[0] is not None, matched_episode_origins=12)
    if not expected_keys or set(rows) != expected_keys:
        raise ValueError('complete four-block metric grid required for every matrix cell')
    evidence = dict(delivery_sha256=sha256_file(archive / 'delivery.json'),
        analysis_sha256=consumed[report_path], paired_block_csv_sha256=consumed[table_path],
        source_comparison=delivery['source_comparison'], source_delivery_sha256=source_digest)
    return budget, seed, report, rows, evidence


def aggregate_prior_matrix(completion_chain, reference_analysis, output, *, staged_inputs=None):
    """Describe the completed two-budget, three-seed matrix using four shared blocks."""
    completion_chain, reference_analysis, output = Path(completion_chain), Path(reference_analysis), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cells = _read(completion_chain).get('cells', [])
    expected = {f'budget_{budget}/seed_{seed}': (budget, seed) for budget in (20, 80) for seed in SEEDS}
    reference = 'budget_20/seed_41'
    if (not isinstance(cells, list) or len(cells) != 5
            or {row.get('cell') for row in cells} != set(expected) - {reference}):
        raise ValueError('completion chain must contain exactly the other five registered matrix cells')
    plan = {row['cell']: row for row in cells}
    plan[reference] = dict(analysis_archive=str(reference_analysis))
    names = [Path(row['analysis_archive']).name for row in plan.values()]
    if len(set(names)) != 6:
        raise ValueError('six distinct analysis archives required')
    all_rows, bindings, budget_bindings = [], {}, {}
    common = None
    for cell in sorted(expected):
        declared = plan[cell]
        original_archive = Path(declared['analysis_archive'])
        archive = original_archive if staged_inputs is None else Path(staged_inputs) / original_archive.name
        source_override = None if staged_inputs is None else archive / 'source_delivery.json'
        budget, seed, report, rows, evidence = _matrix_cell(archive, source_override)
        if ((budget, seed) != expected[cell]
                or ('forecast_archive' in declared
                    and evidence['source_comparison'] != declared['forecast_archive'])):
            raise ValueError('bound budget, seed, or source differs from the declared matrix cell')
        binding = {key: value for key, value in report['training_binding'].items() if key != 'seed'}
        cohort_keys = {'config', 'acquisition_design_sha256', 'acquisition_ledger_sha256', 'dataset_sha256',
            'episode_table_sha256', 'native_source_identity', 'episode_bindings', 'source_revision', 'sources', 'backend'}
        original_digest = report.get('completion_contracts', {}).get('original', {}).get('delivery_sha256')
        if not cohort_keys <= binding.keys() or not isinstance(original_digest, str) or len(original_digest) != 64:
            raise ValueError('complete common cohort and original archive binding required')
        if budget in budget_bindings and budget_bindings[budget] != binding:
            raise ValueError('seed peers within each network budget must share the full training binding')
        budget_bindings[budget] = binding
        # Continuation training has its own source provenance. Compare the
        # measurement/model contract across budgets and retain both source sets.
        identity = dict(training_binding={key: value for key, value in binding.items() if key != 'sources'},
                        configuration=report['configuration'], metrics=report['metrics'],
                        original_delivery_sha256=original_digest,
                        availability={key: row['available'] for key, row in rows.items()})
        if common is None:
            common = identity
        elif common != identity:
            raise ValueError('matrix cells differ in cohort, training configuration, forecast protocol, or availability')
        all_rows.extend(rows[key] for key in sorted(rows))
        bindings[cell] = dict(analysis_archive=str(original_archive), **evidence)
    by_block, summary = [], []
    value_names = ('original_prior', 'fitted_prior', 'fitted_minus_original')
    for budget in (20, 80):
        for horizon in common['configuration']['horizons_ms']:
            for metric in common['metrics']:
                selected = [row for row in all_rows if row['network_budget_epochs'] == budget
                            and row['horizon_ms'] == horizon and row['metric'] == metric]
                block_means = []
                for block in range(14, 18):
                    members = [row for row in selected if row['block'] == block]
                    values = {name: _mean([row[name] for row in members]) for name in value_names}
                    block_means.append(values)
                    by_block.append(dict(network_budget_epochs=budget, block=block, horizon_ms=horizon,
                        metric=metric, **values, available=values['fitted_minus_original'] is not None,
                        independent_blocks=4, training_seeds=3))
                seed_means = [_mean([row['fitted_minus_original'] for row in selected if row['seed'] == seed])
                              for seed in SEEDS]
                available = seed_means[0] is not None
                summary.append(dict(network_budget_epochs=budget, horizon_ms=horizon, metric=metric,
                    **{name + '_mean': _mean([row[name] for row in block_means]) for name in value_names},
                    minimum_seed_mean_difference=min(seed_means) if available else None,
                    maximum_seed_mean_difference=max(seed_means) if available else None,
                    available=available, independent_blocks=4, training_seeds=3, interpretation='development_descriptive'))
    tables = {'forecast_all_seed_blocks.csv': all_rows, 'forecast_seed_averaged_blocks.csv': by_block,
              'forecast_matrix_summary.csv': summary}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    report = dict(status='DESCRIBED_REPLAYED_PRIOR_MATRIX', model='lfads', network_budget_epochs=[20, 80],
        training_seeds=3, training_seed_ids=list(SEEDS), independent_blocks=4, connectivity_instances=1,
        completed_cells=6, configuration=common['configuration'], metrics=common['metrics'],
        source_binding=common['training_binding'], original_delivery_sha256=common['original_delivery_sha256'],
        training_bindings_by_budget={str(budget): binding for budget, binding in budget_bindings.items()},
        source_provenance_changes_between_budgets=budget_bindings[20]['sources'] != budget_bindings[80]['sources'],
        completion_chain_sha256=sha256_file(completion_chain), reference_analysis=str(reference_analysis),
        cell_bindings=bindings, difference='fitted_prior minus original_prior',
        aggregation='equal mean of three seeds within each same block, then equal mean of four blocks; seed range uses each seed mean across those four blocks',
        unavailable_metrics='matched empty metrics retained as unavailable; no imputation or subset selection',
        uncertainty='descriptive development comparison; four shared blocks, not twelve independent seed-block units; no inferential CI/p/q',
        original_prior_reused=True, new_forecast_origins=0, neural_training_updates=0,
        all_unit_metrics_independently_replayed=True, scientific_claim_qualified=False, convergence_established=False,
        formal_test_accessed=False, calibration_outcomes_accessed=False,
        implementation_sha256=sha256_file(Path(__file__)), artifacts={name: sha256_file(output / name) for name in tables})
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--forecasts', type=Path, help='Existing complete three-seed forecast root')
    inputs.add_argument('--prior-comparison', type=Path, help='Completed original_prior/fitted_prior run root')
    inputs.add_argument('--prior-matrix', type=Path, help='Existing completion-chain JSON identifying the other five cells')
    parser.add_argument('--reference-analysis', type=Path, help='Completed reused-original budget_20/seed_41 analysis archive')
    parser.add_argument('--matrix-inputs', type=Path, help='Sparse local archives keyed by original archive basename, with source_delivery.json')
    parser.add_argument('--original-forecast', type=Path, help='Reuse this delivered original-prior forecast directory')
    parser.add_argument('--original-replay', type=Path, help='Independent replay JSON for --original-forecast')
    parser.add_argument('--fit-run', type=Path, help='Optional relocated parent fit proof for a fitted-only candidate')
    parser.add_argument('--decompose-field-errors', action='store_true',
                        help='Partition existing paired field errors; requires --prior-comparison')
    parser.add_argument('--readout-checkpoint', type=Path, help='Reconstruct conditional means using the unchanged selected checkpoint')
    parser.add_argument('--field-view', type=Path, help='Bound train_block_014_familiar_0 source field view')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.prior_matrix is not None and args.reference_analysis is None:
        parser.error('--prior-matrix requires --reference-analysis')
    if (args.reference_analysis is not None or args.matrix_inputs is not None) and args.prior_matrix is None:
        parser.error('matrix reference and staging inputs require --prior-matrix')
    if (args.original_forecast is None) != (args.original_replay is None):
        parser.error('--original-forecast and --original-replay must be supplied together')
    if (args.original_forecast is not None or args.fit_run is not None) and args.prior_comparison is None:
        parser.error('explicit original or fit inputs require --prior-comparison')
    if (args.decompose_field_errors or args.readout_checkpoint is not None or args.field_view is not None) and args.prior_comparison is None:
        parser.error('field error and readout diagnostics require --prior-comparison')
    report = (aggregate_prior_matrix(args.prior_matrix, args.reference_analysis, args.output,
                                    staged_inputs=args.matrix_inputs) if args.prior_matrix is not None
              else analyze(args.forecasts, args.output) if args.forecasts is not None
              else compare_priors(args.prior_comparison, args.output, decompose_field_errors=args.decompose_field_errors,
                                  readout_checkpoint=args.readout_checkpoint, field_view=args.field_view,
                                  original_forecast=args.original_forecast, original_replay=args.original_replay,
                                  fit_run=args.fit_run))
    print(json.dumps({k:v for k,v in report.items() if k not in ('model_seed_bindings','artifacts','source_binding','training_binding','arm_bindings','cell_bindings')}))


if __name__=='__main__':main()

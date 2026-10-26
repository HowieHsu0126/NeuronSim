"""Compare causal field histories on future mean and joint-response statistics.

This deterministic regression diagnostic uses whole development blocks. It is
not a generative simulator, a full-observation oracle, or a formal closure test.
Template/origin cell means condition on the fixed known control programs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .conditional_regression import conditional_prediction
from .provenance import sha256_file
from .trajectory_scores import field_pairs


SCHEMA = 'neural_population_field_history_probe_v1'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_config(config):
    if (config['schema'] != SCHEMA or config['evidence_role'] != 'exploratory_development'
            or config['source_split'] != 'train' or config['formal_training_seeds'] != [41, 42, 43]
            or config['training_rng'] is not None):
        raise ValueError('development-only deterministic probe configuration required')
    fit, selection = config['fit_blocks'], config['selection_blocks']
    if (len(fit) < 3 or not selection or len(set(fit)) != len(fit)
            or len(set(selection)) != len(selection) or set(fit) & set(selection)):
        raise ValueError('distinct whole fitting and selection blocks required')
    if (config['representations'] != ['recent', 'recent_and_history']
            or config['predictors'] != ['linear', 'rbf']
            or config['future_statistics'] != ['mean_rate', 'spatial_variogram',
                'cross_population_variogram', 'temporal_variogram']
            or config['variogram_power'] != 1.0):
        raise ValueError('registered representation and mean/p=1 variogram comparison required')
    grid = config['regularization']
    if not grid or not np.isfinite(grid).all() or min(grid) <= 0 or len(set(grid)) != len(grid):
        raise ValueError('distinct positive finite regularization values required')
    for key in ['recent_ms', 'history_ms', 'history_bin_ms', 'temporal_lag_ms']:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('positive finite observation windows required')


def bin_count(milliseconds, bin_ms):
    value = milliseconds/bin_ms
    if not np.isfinite(value) or value <= 0 or value != int(value):
        raise ValueError('observation windows must align with source bins')
    return int(value)


def window_features(rates, origin_ms, bin_ms, config):
    """Extract only bins ending at or before the prediction boundary."""
    boundary = bin_count(origin_ms, bin_ms)
    recent = bin_count(config['recent_ms'], bin_ms)
    history = bin_count(config['history_ms'], bin_ms)
    width = bin_count(config['history_bin_ms'], bin_ms)
    if history % width or min(boundary-recent, boundary-history) < 0 or boundary > len(rates):
        raise ValueError('complete aligned causal field histories required')
    r = rates[boundary-recent:boundary].mean(axis=0)
    h = rates[boundary-history:boundary].reshape(history//width, width, rates.shape[1]).mean(axis=1)
    return {'recent': r, 'recent_and_history': np.concatenate([r, h.ravel()])}


def future_statistics(rates, origin_ms, horizon_ms, bin_ms, pairs, config):
    start, length = bin_count(origin_ms, bin_ms), bin_count(horizon_ms, bin_ms)
    lag = bin_count(config['temporal_lag_ms'], bin_ms)
    if start+length > len(rates) or lag >= length:
        raise ValueError('complete future support and a shorter temporal lag required')
    y = rates[start:start+length]
    result = {'mean_rate': y.mean(axis=0)}
    for name, ij in pairs.items():
        result[name+'_variogram'] = np.abs(y[:, ij[:, 0]]-y[:, ij[:, 1]]).mean(axis=0)
    result['temporal_variogram'] = np.abs(y[:-lag]-y[lag:]).mean(axis=0)
    return result


def build(source, config, output):
    source, output = Path(source), Path(output)
    validate_config(config)
    if output.exists():
        raise FileExistsError(output)
    acquisition, fields = source/'train', source/'train_fields'
    design, dataset = read_json(acquisition/'design.json'), read_json(fields/'dataset.json')
    if (design['acquired_split'] != 'train' or dataset['status'] != 'BUILT'
            or dataset['split'] != 'train' or dataset['test_outcomes_accessed']
            or dataset['completed_episodes'] != config['expected_episodes']
            or dataset['cortical_units'] != [config['expected_units']]
            or dataset['refractory_violations'] or dataset['control_delivery_violations']
            or dataset['design_sha256'] != sha256_file(acquisition/'design.json')
            or design['config']['execution']['expected_poisson_rng_family'] != config['expected_source_rng']
            or dataset['episode_table_sha256'] != sha256_file(fields/'episodes.csv')):
        raise ValueError('complete source-bound Philox development fields required')
    with (fields/'episodes.csv').open(encoding='utf-8') as stream:
        records = {row['case_id']: row for row in csv.DictReader(stream)}
    cases = sorted(design['cases'], key=lambda x:(x['block'], x['template_id']))
    blocks = config['fit_blocks']+config['selection_blocks']
    expected = set(itertools.product(blocks, config['templates']))
    if (len(cases) != len(expected) or len(records) != len(expected)
            or {(c['block'], c['template_id']) for c in cases} != expected):
        raise ValueError('all registered block/template episodes required')
    bin_ms = dataset['settings']['bin_ms']
    rows, targets, features, bindings = [], [], {key:[] for key in config['representations']}, []
    reference, programs, total_counts, target_slices = None, {}, 0, None
    for case in cases:
        name = case['case_id']
        if (case['split'] != 'train' or Path(name).name != name or name not in records):
            raise ValueError('source episode identity or split mismatch')
        directory, row = fields/name, records[name]
        summary = read_json(directory/'summary.json')
        if (sha256_file(directory/'summary.json') != row['field_summary_sha256']
                or summary['episode_report_sha256'] != row['raw_report_sha256']
                or summary['status'] != 'DESCRIBED' or summary['cortical_refractory_violations']
                or int(row['block']) != case['block'] or row['template_id'] != case['template_id']
                or int(row['cortical_units']) != config['expected_units']):
            raise ValueError('field table, source identity or summary changed')
        path = directory/'spatial_fields.npz'
        if sha256_file(path) != summary['artifacts']['spatial_fields.npz']:
            raise ValueError('field arrays changed')
        with np.load(path, allow_pickle=False) as archive:
            a = {key:archive[key] for key in archive.files}
        identities = {k:a[k] for k in ['source_unit_ids', 'unit_field_index', 'field_ids',
            'total_unit_count', 'bin_edges_ticks', 'dt_ms']}
        counts, units = a['count_sum'], a['total_unit_count']
        if (counts.ndim != 2 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)
                or len(a['source_unit_ids']) != config['expected_units']
                or len(np.unique(a['source_unit_ids'])) != config['expected_units']
                or np.any(units <= 0) or not np.all(a['complete'])
                or not np.array_equal(a['observed_unit_count'], np.broadcast_to(units, counts.shape))
                or not np.array_equal(np.bincount(a['unit_field_index'], minlength=len(units)), units)
                or a['bin_edges_ticks'][0] != 0
                or not np.all(np.diff(a['bin_edges_ticks'])*a['dt_ms'] == bin_ms)):
            raise ValueError('complete identified fields and native clock required')
        if reference is None:
            reference = identities
            definitions = [json.loads(str(f)) for f in a['field_ids']]
            pairs = field_pairs([r[0] for r in definitions], [r[2] for r in definitions], [r[1] for r in definitions])
        elif any(not np.array_equal(reference[k], value) for k, value in identities.items()):
            raise ValueError('field/unit/clock identities differ across episodes')
        if int(counts.sum()) != summary['cortical_events_in_window']:
            raise ValueError('field totals differ from source-bound event counts')
        rates = counts/(units[None, :]*(bin_ms/1000))
        if not np.isfinite(rates).all() or not np.allclose(rates, a['observed_rate_hz'], atol=1e-12, rtol=1e-12):
            raise ValueError('count/Hz observation mapping changed')
        program = case['settings']['pulse_control']
        template = case['template_id']
        if template in programs and programs[template] != program:
            raise ValueError('template cell does not identify one fixed control program')
        programs[template] = program
        for origin_index, origin in enumerate(config['origins_ms']):
            x = window_features(rates, origin, bin_ms, config)
            for key in features:
                features[key].append(x[key])
            parts, slices, offset = [], [], 0
            for horizon in config['horizons_ms']:
                stats = future_statistics(rates, origin, horizon, bin_ms, pairs, config)
                for family in config['future_statistics']:
                    y = stats[family]
                    slices.append(dict(family=family, horizon_ms=horizon, start=offset,
                        stop=offset+len(y), statistic_units='Hz', mse_units='Hz^2'))
                    offset += len(y)
                    parts.append(y)
            if target_slices is not None and slices != target_slices:
                raise ValueError('target projection changed')
            target_slices = slices
            targets.append(np.concatenate(parts))
            rows.append(dict(case_id=name, block=case['block'], template_id=template,
                origin_ms=origin, cell=config['templates'].index(template)*len(config['origins_ms'])+origin_index))
        total_counts += int(counts.sum())
        bindings.append(dict(case_id=name, field_summary_sha256=row['field_summary_sha256'],
            spatial_fields_sha256=summary['artifacts']['spatial_fields.npz']))
    if total_counts != dataset['cortical_events']:
        raise ValueError('complete cohort event coverage mismatch')
    arrays = {k:np.asarray(v) for k,v in features.items()}
    arrays.update(targets=np.asarray(targets), blocks=np.array([r['block'] for r in rows]),
        cells=np.array([r['cell'] for r in rows]), field_ids=reference['field_ids'],
        spatial_pairs=pairs['spatial'], cross_population_pairs=pairs['cross_population'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'features.npz', **arrays)
    report = dict(status='BUILT_DEVELOPMENT', config=config, rows=rows, target_slices=target_slices,
        source_dataset_sha256=sha256_file(fields/'dataset.json'), source_bindings=bindings,
        source_design_sha256=sha256_file(acquisition/'design.json'), source_field_counts=total_counts,
        feature_dimensions={key:arrays[key].shape[1] for key in features},
        unit_count=config['expected_units'], field_count=len(reference['field_ids']),
        source_bin_ms=bin_ms, features_sha256=sha256_file(output/'features.npz'),
        implementation_sha256=sha256_file(Path(__file__)),
        source_fidelity_qualified=False, formal_training_completed=False,
        calibration_outcomes_accessed=False, test_outcomes_accessed=False)
    write_json(output/'features.json', report)
    return report


def fit_predictions(arrays, config, slices):
    """Select penalties inside fit blocks, then predict every selection row."""
    y, blocks, cells = arrays['targets'], arrays['blocks'], arrays['cells']
    fit = np.isin(blocks, config['fit_blocks'])
    selection = np.isin(blocks, config['selection_blocks'])
    if not np.all(fit ^ selection):
        raise ValueError('every row must belong to exactly one registered block set')
    models = [('cells', 'recent', 'cells')]+[(r+'_'+k,r,k)
        for r in config['representations'] for k in config['predictors']]
    predictions = np.empty((len(models), int(selection.sum()), y.shape[1]))
    choices, inner_rows = [], []
    for model_index, (name, representation, kind) in enumerate(models):
        x = arrays[representation]
        grid = [1.0] if kind == 'cells' else config['regularization']
        losses = np.zeros((len(grid), len(slices)))
        for ai, alpha in enumerate(grid):
            for block in config['fit_blocks']:
                inner, valid = fit & (blocks != block), blocks == block
                pred, _ = conditional_prediction(x[inner], y[inner], cells[inner], x[valid], cells[valid], kind, alpha)
                for si, spec in enumerate(slices):
                    take = slice(spec['start'], spec['stop'])
                    error = float(np.mean((pred[:,take]-y[valid,take])**2))
                    losses[ai,si] += error/len(config['fit_blocks'])
                    inner_rows.append(dict(model=name, alpha=alpha, fit_validation_block=block,
                        family=spec['family'], horizon_ms=spec['horizon_ms'], mse=error))
        selected = np.argmin(losses, axis=0)
        for ai in np.unique(selected):
            pred, diagnostic = conditional_prediction(x[fit], y[fit], cells[fit], x[selection], cells[selection], kind, grid[ai])
            for si in np.flatnonzero(selected == ai):
                spec = slices[si]
                take = slice(spec['start'], spec['stop'])
                predictions[model_index,:,take] = pred[:,take]
                choices.append(dict(model=name, representation=representation, predictor=kind,
                    family=spec['family'], horizon_ms=spec['horizon_ms'], alpha=grid[ai],
                    fitting_cv_mse=float(losses[ai,si]), **diagnostic))
    return predictions, [m[0] for m in models], choices, inner_rows, selection


def analyze(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = read_json(dataset/'features.json')
    if report['status'] != 'BUILT_DEVELOPMENT' or sha256_file(dataset/'features.npz') != report['features_sha256']:
        raise ValueError('unchanged development feature archive required')
    config, slices = report['config'], report['target_slices']
    validate_config(config)
    with np.load(dataset/'features.npz', allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    started = time.perf_counter()
    predictions, names, choices, inner, selection = fit_predictions(arrays, config, slices)
    truth, blocks = arrays['targets'][selection], arrays['blocks'][selection]
    by_block, summaries, changes = [], [], []
    errors = {}
    for mi, name in enumerate(names):
        for si, spec in enumerate(slices):
            take = slice(spec['start'], spec['stop'])
            per_row = np.mean((predictions[mi,:,take]-truth[:,take])**2, axis=1)
            errors[(name,si)] = []
            for block in config['selection_blocks']:
                value = float(per_row[blocks == block].mean())
                errors[(name,si)].append(value)
                by_block.append(dict(model=name, family=spec['family'], horizon_ms=spec['horizon_ms'],
                    block=block, mse=value, rows=int(np.sum(blocks == block))))
            summaries.append(dict(model=name, family=spec['family'], horizon_ms=spec['horizon_ms'],
                mean_block_mse=float(np.mean(errors[(name,si)])), selection_blocks=len(config['selection_blocks']),
                rows=len(truth), dimensions=spec['stop']-spec['start'], units=spec['mse_units']))
    for si, spec in enumerate(slices):
        for kind in config['predictors']:
            a = np.array(errors[('recent_'+kind,si)])
            b = np.array(errors[('recent_and_history_'+kind,si)])
            changes.append(dict(predictor=kind, family=spec['family'], horizon_ms=spec['horizon_ms'],
                history_minus_recent_mse=float((b-a).mean()),
                relative_mse_change=float(b.mean()/a.mean()-1) if a.mean() else None,
                blocks_improved=int(np.sum(b<a)), blocks=len(a)))
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz', predictions=predictions, model_names=np.asarray(names),
        target=truth, blocks=blocks, cells=arrays['cells'][selection], selection_indices=np.flatnonzero(selection))
    for name, rows in [('penalty_selection.csv',choices),('inner_validation.csv',inner),
            ('errors_by_block.csv',by_block),('model_summary.csv',summaries),('history_effects.csv',changes)]:
        write_rows(output/name, rows)
    result = dict(status='ANALYZED_DEVELOPMENT', created_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started, fitting_blocks=config['fit_blocks'],
        selection_blocks=config['selection_blocks'], selection_rows=len(truth), models=names,
        deterministic_regression=True, stochastic_training_seeds_executed=[],
        interpretation=config['interpretation'], full_observation_oracle=False,
        state_sufficiency_proven=False, neuroscience_finding_qualified=False,
        formal_training_completed=False, calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        feature_report_sha256=sha256_file(dataset/'features.json'),
        artifacts={p.name:sha256_file(p) for p in output.iterdir()},
        implementation_sha256=sha256_file(Path(__file__)),
        regression_sha256=sha256_file(Path(__file__).with_name('conditional_regression.py')))
    write_json(output/'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    for name in ['source', 'config', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('analyze')
    for name in ['dataset', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'build':
        result = build(args.source, yaml.safe_load(args.config.read_text(encoding='utf-8')), args.output)
    else:
        result = analyze(args.dataset, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows', 'source_bindings']}, indent=2))


if __name__ == '__main__':
    main()

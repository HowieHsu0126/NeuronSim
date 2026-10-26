"""Independently replay field-history features, selected fits and score tables.

Uses integer count sums for time averages, explicit cell-wise kernel centering,
SciPy distances and a Cholesky solve. It does not call the producer's feature,
target, regression, selection or scoring functions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from .provenance import sha256_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def independent_prediction(x, y, cells, query, query_cells, kind, alpha):
    labels = sorted(set(cells.tolist()))
    groups = [np.flatnonzero(cells == label) for label in labels]
    queries = [np.flatnonzero(query_cells == label) for label in labels]
    center, query_center, residual = x.copy(), query.copy(), y.copy()
    means = np.zeros((len(query), y.shape[1]))
    for group, selected in zip(groups, queries):
        average = x[group].mean(axis=0)
        center[group] -= average
        query_center[selected] -= average
        ym = y[group].mean(axis=0)
        residual[group] -= ym
        means[selected] = ym
    active = np.zeros(x.shape[1], dtype=bool)
    for group in groups:
        active |= np.ptp(x[group], axis=0) > 0
    if kind == 'cells' or not active.any():
        return means
    scale = np.std(center, axis=0)[active]
    z, q = center[:,active]/scale, query_center[:,active]/scale
    if kind == 'linear':
        kernel, cross = z @ z.T, q @ z.T
    elif kind == 'rbf':
        kernel = np.exp(-cdist(z,z,'sqeuclidean')/(2*z.shape[1]))
        cross = np.exp(-cdist(q,z,'sqeuclidean')/(2*z.shape[1]))
    else:
        raise ValueError('unknown independent predictor')
    # First subtract the fitting-row mean for each conditioning cell; then
    # subtract each fitting-column cell mean. Queries use fitting-row means.
    centered, query_kernel = kernel.copy(), cross.copy()
    for group, selected in zip(groups,queries):
        km = kernel[group].mean(axis=0)
        centered[group] -= km
        query_kernel[selected] -= km
    for group in groups:
        centered[:,group] -= centered[:,group].mean(axis=1)[:,None]
        query_kernel[:,group] -= query_kernel[:,group].mean(axis=1)[:,None]
    regularized = centered + len(x)*alpha*np.eye(len(x))
    # Averaging cancels only numerical antisymmetry, not a scientific term.
    regularized = (regularized+regularized.T)/2
    coefficients = cho_solve(cho_factor(regularized,lower=True),residual)
    return means + query_kernel @ coefficients


def replay(source, dataset, analysis, output):
    source, dataset, analysis, output = map(Path,(source,dataset,analysis,output))
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    feature = read(dataset/'features.json')
    producer = read(analysis/'analysis.json')
    if (producer['status'] != 'ANALYZED_DEVELOPMENT'
            or sha256_file(dataset/'features.json') != producer['feature_report_sha256']
            or sha256_file(dataset/'features.npz') != feature['features_sha256']
            or sha256_file(source/'train_fields/dataset.json') != feature['source_dataset_sha256']):
        raise ValueError('completed source-bound feature and analysis artifacts required')
    for name,digest in producer['artifacts'].items():
        if sha256_file(analysis/name) != digest:
            raise ValueError('analysis artifact changed: '+name)
    config, specs = feature['config'], feature['target_slices']
    atol, rtol = config['score_replay_atol'], config['score_replay_rtol']
    compared, largest = {}, {}
    def compare(key, actual, expected):
        a,b = np.asarray(actual),np.asarray(expected)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('invalid replay arrays: '+key)
        if not np.allclose(a,b,atol=atol,rtol=rtol):
            raise ValueError('independent replay mismatch: '+key)
        compared[key] = compared.get(key,0)+int(a.size)
        largest[key] = max(largest.get(key,0.),float(np.max(np.abs(a-b))) if a.size else 0.)
    with np.load(dataset/'features.npz',allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    bin_ms = feature['source_bin_ms']
    definitions = [json.loads(str(v)) for v in arrays['field_ids']]
    nx,ny = max(d[2] for d in definitions)+1,max(d[1] for d in definitions)+1
    # Check adjacency without using the producer's dictionary lookup.
    for name,ij in [('spatial',arrays['spatial_pairs']),('cross_population',arrays['cross_population_pairs'])]:
        found = set()
        for a in range(len(definitions)):
            p,y,x = definitions[a]
            for b in range(a+1,len(definitions)):
                q,v,u = definitions[b]
                spatial = p==q and ((x==u and (y-v)%ny in (1,ny-1)) or (y==v and (x-u)%nx in (1,nx-1)))
                cross = p!=q and (y,x)==(v,u)
                if (name=='spatial' and spatial) or (name=='cross_population' and cross):
                    found.add((a,b))
        if {tuple(sorted(pair)) for pair in ij.tolist()} != found or len(ij)!=len(found):
            raise ValueError('independent field adjacency differs')
    grouped = {}
    for i,row in enumerate(feature['rows']):
        grouped.setdefault(row['case_id'],[]).append((i,row))
    bindings = {row['case_id']:row for row in feature['source_bindings']}
    for case, selected in grouped.items():
        path = source/'train_fields'/case/'spatial_fields.npz'
        if sha256_file(path) != bindings[case]['spatial_fields_sha256']:
            raise ValueError('source count artifact changed')
        with np.load(path,allow_pickle=False) as raw:
            count,units = raw['count_sum'],raw['total_unit_count']
        for i,row in selected:
            boundary = int(row['origin_ms']/bin_ms)
            width,history = int(config['history_bin_ms']/bin_ms),int(config['history_ms']/bin_ms)
            recent = int(config['recent_ms']/bin_ms)
            r = count[boundary-recent:boundary].sum(axis=0)*1000/(units*config['recent_ms'])
            h = np.array([count[j:j+width].sum(axis=0)*1000/(units*config['history_bin_ms'])
                for j in range(boundary-history,boundary,width)])
            compare('recent_features',r,arrays['recent'][i])
            compare('augmented_features',np.concatenate([r,h.ravel()]),arrays['recent_and_history'][i])
            for spec in specs:
                length = int(spec['horizon_ms']/bin_ms)
                c = count[boundary:boundary+length]
                if spec['family']=='mean_rate':
                    value = c.sum(axis=0)*1000/(units*spec['horizon_ms'])
                else:
                    rate = c/units[None,:]*1000/bin_ms
                    if spec['family']=='temporal_variogram':
                        lag = int(config['temporal_lag_ms']/bin_ms)
                        value = np.sum(np.abs(rate[lag:]-rate[:-lag]),axis=0)/(length-lag)
                    else:
                        name = spec['family'].removesuffix('_variogram')
                        ij = arrays[name+'_pairs']
                        value = np.array([np.sum(np.abs(rate[:,a]-rate[:,b]))/length for a,b in ij])
                compare('future_targets',value,arrays['targets'][i,spec['start']:spec['stop']])
    with np.load(analysis/'predictions.npz',allow_pickle=False) as data:
        saved = {k:data[k] for k in data.files}
    fit = np.isin(arrays['blocks'],config['fit_blocks'])
    selected = np.isin(arrays['blocks'],config['selection_blocks'])
    if not np.array_equal(saved['selection_indices'],np.flatnonzero(selected)):
        raise ValueError('selection row identity changed')
    if (not np.array_equal(saved['blocks'],arrays['blocks'][selected])
            or not np.array_equal(saved['cells'],arrays['cells'][selected])):
        raise ValueError('selection block/cell identity changed')
    compare('selection_truth',saved['target'],arrays['targets'][selected])
    choices = rows(analysis/'penalty_selection.csv')
    inner = rows(analysis/'inner_validation.csv')
    cache, predictions = {}, {}
    for choice in choices:
        name,rep,kind,alpha = (choice[k] for k in ['model','representation','predictor','alpha'])
        key = (name,alpha)
        if key not in predictions:
            predictions[key] = independent_prediction(arrays[rep][fit],arrays['targets'][fit],arrays['cells'][fit],
                arrays[rep][selected],arrays['cells'][selected],kind,float(alpha))
        spec = next(s for s in specs if s['family']==choice['family'] and s['horizon_ms']==float(choice['horizon_ms']))
        mi = list(saved['model_names']).index(name)
        take = slice(spec['start'],spec['stop'])
        compare('selected_predictions',predictions[key][:,take],saved['predictions'][mi,:,take])
        grid = [1.] if kind=='cells' else config['regularization']
        losses = []
        for alpha in grid:
            scores = []
            for block in config['fit_blocks']:
                key = (name,float(alpha),block)
                if key not in cache:
                    training,valid = fit & (arrays['blocks']!=block),arrays['blocks']==block
                    cache[key] = independent_prediction(arrays[rep][training],arrays['targets'][training],arrays['cells'][training],
                        arrays[rep][valid],arrays['cells'][valid],kind,alpha)
                actual = arrays['targets'][arrays['blocks']==block,take]
                mse = np.mean((cache[key][:,take]-actual)**2)
                row = next(r for r in inner if r['model']==name and float(r['alpha'])==alpha
                    and int(r['fit_validation_block'])==block and r['family']==spec['family']
                    and float(r['horizon_ms'])==spec['horizon_ms'])
                compare('inner_cv_losses',mse,float(row['mse']))
                scores.append(mse)
            losses.append(np.mean(scores))
        if grid[int(np.argmin(losses))] != float(choice['alpha']):
            raise ValueError('independent penalty selection changed')
        compare('selected_cv_losses',min(losses),float(choice['fitting_cv_mse']))
    block_errors = {}
    for row in rows(analysis/'errors_by_block.csv'):
        mi = list(saved['model_names']).index(row['model'])
        spec = next(s for s in specs if s['family']==row['family'] and s['horizon_ms']==float(row['horizon_ms']))
        mask = saved['blocks']==int(row['block'])
        take = slice(spec['start'],spec['stop'])
        value = np.mean((saved['predictions'][mi,mask,take]-saved['target'][mask,take])**2)
        compare('block_mse',value,float(row['mse']))
        block_errors.setdefault((row['model'],row['family'],row['horizon_ms']),[]).append(value)
    for row in rows(analysis/'model_summary.csv'):
        value = np.mean(block_errors[(row['model'],row['family'],row['horizon_ms'])])
        compare('summary_mse',value,float(row['mean_block_mse']))
    for row in rows(analysis/'history_effects.csv'):
        base = np.array(block_errors[('recent_'+row['predictor'],row['family'],row['horizon_ms'])])
        history = np.array(block_errors[('recent_and_history_'+row['predictor'],row['family'],row['horizon_ms'])])
        compare('history_effects',[(history-base).mean(),history.mean()/base.mean()-1,int(np.sum(history<base))],
            [float(row['history_minus_recent_mse']),float(row['relative_mse_change']),int(row['blocks_improved'])])
    result = dict(status='INDEPENDENT_REPLAY_PASS',elapsed_seconds=time.perf_counter()-started,
        compared_values=compared,max_absolute_difference=largest,atol=atol,rtol=rtol,
        analysis_sha256=sha256_file(analysis/'analysis.json'),implementation_sha256=sha256_file(Path(__file__)),
        source_fidelity_qualified=False,neuroscience_finding_qualified=False,
        formal_training_completed=False,calibration_outcomes_accessed=False,test_outcomes_accessed=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','dataset','analysis','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(replay(args.source,args.dataset,args.analysis,args.output),indent=2))


if __name__=='__main__':
    main()

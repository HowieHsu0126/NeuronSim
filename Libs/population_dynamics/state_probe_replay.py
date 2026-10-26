"""Independent raw-event feature and augmented-system replay of state probes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import sha256_file


def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def fit(x,y,cells,xt,ct,kind,alpha):
    levels=sorted(set(cells.tolist()));means=[];keep=np.zeros(x.shape[1],dtype=bool)
    for c in levels:
        subset=x[cells==c];means.append(subset.mean(0));keep|=np.any(subset!=subset[0],axis=0)
    means=np.array(means);ci=np.array([levels.index(c) for c in cells]);ti=np.array([levels.index(c) for c in ct])
    yc=np.array([y[cells==c].mean() for c in levels])
    if kind=='cells' or not keep.any():return yc[ti]
    centered=x-means[ci];sd=centered.std(0);z=centered[:,keep]/sd[keep];zt=(xt-means[ti])[:,keep]/sd[keep]
    if kind=='linear':k=z@z.T;kt=zt@z.T
    else:
        k=np.array([[np.exp(-np.dot(a-b,a-b)/(2*z.shape[1])) for b in z] for a in z])
        kt=np.array([[np.exp(-np.dot(a-b,a-b)/(2*z.shape[1])) for b in z] for a in zt])
    cmat=np.eye(len(levels))[ci];tmat=np.eye(len(levels))[ti];n=len(x)
    system=np.block([[k+alpha*n*np.eye(n),cmat],[cmat.T,np.zeros((len(levels),len(levels)))]])
    solution=np.linalg.solve(system,np.concatenate([y,np.zeros(len(levels))]))
    return kt@solution[:n]+tmat@solution[n:]


def verify(dataset,analysis,acquisition,boundaries,graph,legacy,output):
    meta=json.loads((dataset/'features.json').read_text());config=meta['config'];cfg=config['features']
    if sha256_file(dataset/'representations.npz')!=meta['arrays_sha256']:raise ValueError('representation artifact changed')
    source=json.loads((graph/'manifest.json').read_text());names=source['resolved_source_network']['populations'][:-1]
    extent=source['resolved_source_network']['extent'];naxis=cfg['spatial_cells_per_axis'];edges=np.linspace(-extent/2,extent/2,naxis+1)
    data=arrays(dataset/'representations.npz');pred=arrays(analysis/'predictions.npz');checks={};max_feature_error=0.
    for row_index,row in enumerate(meta['rows']):
        case=acquisition/row['omitted_case'];events=arrays(case/'spikes.npz');dt=float(events['dt_ms'])
        cutoff=round((row['focal_arrival_ms']-source['resolved_source_network']['th_delay_pulse_generator'])/dt)
        ids=events['unit_ids'];labels=events['populations'];position=events['positions_mm'];senders=events['spike_unit_ids'];time=events['spike_ticks']
        idx={int(u):j for j,u in enumerate(ids)};event_index=np.array([idx[int(u)] for u in senders]);event_labels=labels[event_index]
        past=time<cutoff;features=[]
        for window in cfg['rate_windows_ms']:
            for name in names:
                take=past&(time>=cutoff-round(window/dt))&(event_labels==name)
                features.append(np.count_nonzero(take)/np.count_nonzero(labels==name)/(window/1000))
        rates=np.array(features)
        for begin in np.arange(-cfg['temporal_window_ms'],0,cfg['temporal_bin_ms']):
            for name in names:
                take=(time>=cutoff+round(begin/dt))&(time<cutoff+round((begin+cfg['temporal_bin_ms'])/dt))&(event_labels==name)
                features.append(np.count_nonzero(take)/np.count_nonzero(labels==name)/(cfg['temporal_bin_ms']/1000))
        for window in cfg['spatial_windows_ms']:
            for name in sorted(names):
                take=past&(time>=cutoff-round(window/dt))&(event_labels==name)
                xy=position[event_index[take]].copy();xy[xy==extent/2]=-extent/2
                count=np.histogram2d(xy[:,1],xy[:,0],bins=(edges,edges))[0]
                features.extend((count/(window/1000)/(extent/naxis)**2).ravel())
        for window in cfg['spike_age_windows_ms']:
            for name in names:
                take=past&(time>=cutoff-round(window/dt))&(event_labels==name)
                features.append(len(np.unique(senders[take]))/np.count_nonzero(labels==name))
        observed=np.array(features);prefix=boundaries/row['omitted_case'];terminal=arrays(prefix/'terminal_states.npz');initial=arrays(prefix/'initial_parameters.npz')
        cortex_labels=labels[labels!='TC'];gap=initial['Theta_rel']-terminal['V_m_rel'];refractory=terminal['refractory_step']
        extra=[]
        for name in names:
            select=cortex_labels==name;g=gap[select]
            extra.extend([g.mean(),np.sqrt(np.mean((g-g.mean())**2))]);extra.extend(np.percentile(g,100*np.array(cfg['voltage_gap_quantiles'])))
            extra.extend([np.count_nonzero(g<=v)/len(g) for v in cfg['voltage_gap_thresholds_mv']])
            for key in ['I_syn_ex','I_syn_in']:
                values=terminal[key][select];extra.extend([values.mean(),np.sqrt(np.mean((values-values.mean())**2))])
            extra.extend([np.count_nonzero(refractory[select]>0)/np.count_nonzero(select),np.mean(refractory[select])*dt])
        for name,expected in [('R',rates),('S',observed),('P',np.concatenate([observed,extra]))]:
            error=float(np.max(np.abs(data[name][row_index]-expected)));max_feature_error=max(max_feature_error,error)
            checks[f'row_{row_index}_{name}_independent_features']=bool(np.allclose(data[name][row_index],expected,rtol=1e-12,atol=1e-12))
    previous=arrays(legacy/'state_probe_arrays.npz')
    checks['original_rate_features_exact']=np.array_equal(data['R'],previous['features'])
    checks['original_outcomes_exact']=np.array_equal(data['target'],previous['target'])
    unique=sorted(set(data['blocks'].tolist()));replayed=np.empty_like(pred['predictions']);selected=[]
    for column,name in enumerate(pred['model_names']):
        rep,kind=('R','cells') if name=='cells' else str(name).split('_');x=data[rep];y=data['target'];c=data['cells'];b=data['blocks']
        grid=[1.] if kind=='cells' else config['predictors'][kind+'_regularization']
        for block in unique:
            training=b!=block;testing=~training;scores=[]
            for alpha in grid:
                losses=[]
                for vb in [k for k in unique if k!=block]:
                    inside=training&(b!=vb);validation=b==vb
                    q=fit(x[inside],y[inside],c[inside],x[validation],c[validation],kind,alpha);losses.append(np.mean((q-y[validation])**2))
                scores.append(float(np.mean(losses)))
            alpha=grid[int(np.argmin(scores))];selected.append(dict(model=str(name),block=int(block),alpha=alpha))
            replayed[testing,column]=fit(x[training],y[training],c[training],x[testing],c[testing],kind,alpha)
    error=float(np.max(np.abs(replayed-pred['predictions'])))
    checks['all_nested_predictions_independent_augmented_solve']=bool(np.allclose(replayed,pred['predictions'],rtol=1e-9,atol=1e-9))
    import csv
    with (analysis/'prediction_by_block.csv').open() as f:registered={(r['model'],int(r['block'])):float(r['selected_alpha']) for r in csv.DictReader(f)}
    checks['all_selected_settings_exact']=all(registered[(r['model'],r['block'])]==r['alpha'] for r in selected)
    checks['original_cell_and_linear_cv_replayed']=bool(np.allclose(pred['predictions'][:,:2],previous['predictions'],rtol=1e-11,atol=1e-11))
    result=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks,feature_max_absolute_delta=max_feature_error,
        prediction_max_absolute_delta=error,paired_conditions=len(meta['rows']),independent_blocks=len(unique),
        implementation_sha256=sha256_file(Path(__file__)),scientific_reference_qualified=False,neuroscience_finding_qualified=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','analysis','acquisition','boundaries','graph','legacy','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=verify(a.dataset,a.analysis,a.acquisition,a.boundaries,a.graph,a.legacy,a.output)
    print(json.dumps(r,indent=2));raise SystemExit(int(r['status']!='PASS'))


if __name__=='__main__':main()

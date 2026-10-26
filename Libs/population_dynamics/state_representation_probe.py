"""Development comparison of rate, richer observed, and privileged field states.

R/S/P are candidate diagnostics, not a qualified neural simulator. Predictions
retain unpenalized factorial cells; all transformations and selection use whole
training blocks. Cross-validation errors do not supply independent replicates.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .conditional_regression import conditional_prediction
import yaml

from .nest_gpu_network import ticks
from .provenance import sha256_file
from .pulse_history_analysis import _verified_events
from .spatial_field import PeriodicPopulationGrid


def read_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def observed_features(data, event_indices, cutoff, dt, names, extent, config):
    ids,labels,xy=(data[k] for k in ['unit_ids','populations','positions_mm'])
    cortical=labels!='TC';events=data['spike_ticks'];cortical_events=cortical[event_indices]
    lookup={name:i for i,name in enumerate(names)};pop=np.array([lookup.get(str(name),-1) for name in labels])[event_indices]
    sizes=np.array([np.sum(labels==name) for name in names]);n_pop=len(names)
    if np.any(sizes==0):raise ValueError('all declared cortical populations required')
    features=[];feature_names=[]
    for window in config['rate_windows_ms']:
        width=int(ticks([window],dt)[0]);take=cortical_events&(events>=cutoff-width)&(events<cutoff)
        features.extend(np.bincount(pop[take],minlength=n_pop)/sizes/(window/1000))
        feature_names.extend([f'rate_hz_{name}_{window:g}ms' for name in names])
    rate=np.array(features);rate_names=np.array(feature_names)
    width=int(ticks([config['temporal_window_ms']],dt)[0]);step=int(ticks([config['temporal_bin_ms']],dt)[0])
    if width<=0 or step<=0 or width%step or cutoff<width:raise ValueError('complete aligned temporal history required')
    take=cortical_events&(events>=cutoff-width)&(events<cutoff);bin_id=(events[take]-(cutoff-width))//step
    counts=np.bincount(bin_id*n_pop+pop[take],minlength=width//step*n_pop).reshape(-1,n_pop)
    features.extend((counts/sizes/(step*dt/1000)).ravel())
    for j in range(len(counts)):
        feature_names.extend([f'temporal_rate_hz_{name}_{(width-j*step)*dt:g}to{(width-(j+1)*step)*dt:g}ms' for name in names])
    observer=PeriodicPopulationGrid.from_positions(ids[cortical],labels[cortical],xy[cortical],extent_mm=extent,
        cells_per_axis=config['spatial_cells_per_axis'],definition='source cortical population and spatial cell')
    if not np.array_equal(observer.mapping.unit_ids,ids[cortical]):raise ValueError('spatial unit mapping changed order')
    group_by_unit=np.full(len(ids),-1,dtype=int);group_by_unit[cortical]=observer.mapping.group_index
    event_group=group_by_unit[event_indices];n_fields=len(observer.mapping.field_ids)
    for window in config['spatial_windows_ms']:
        width=int(ticks([window],dt)[0]);take=cortical_events&(events>=cutoff-width)&(events<cutoff)
        counts=np.bincount(event_group[take],minlength=n_fields)
        features.extend(counts/(window/1000)/observer.cell_area_mm2)
        feature_names.extend([f'spatial_density_hz_mm2_{field}_{window:g}ms' for field in observer.mapping.field_ids])
    last=np.full(len(ids),-1,dtype=np.int64);take=cortical_events&(events<cutoff)
    np.maximum.at(last,event_indices[take],events[take])
    for window in config['spike_age_windows_ms']:
        width=int(ticks([window],dt)[0]);recent=(last>=cutoff-width)&(last<cutoff)
        features.extend([np.mean(recent[labels==name]) for name in names])
        feature_names.extend([f'fraction_spiked_{name}_{window:g}ms' for name in names])
    return rate,rate_names,np.array(features),np.array(feature_names)


def privileged_features(prefix, data, names, config):
    phase=json.loads((prefix/'prefix_alignment.json').read_text());capture=json.loads((prefix/'boundary_capture.json').read_text())
    if phase['status']!='PASS' or not phase['terminal_tick']<phase['emission_tick']:
        raise ValueError('causal validated terminal phase required')
    if sha256_file(prefix/'terminal_states.npz')!=capture['terminal_states_sha256']:
        raise ValueError('terminal state artifact changed')
    terminal=read_arrays(prefix/'terminal_states.npz');initial=read_arrays(prefix/'initial_parameters.npz')
    labels=data['populations'][data['populations']!='TC'];ids=data['unit_ids'][data['populations']!='TC']
    if not np.array_equal(terminal['unit_ids'],ids) or not np.array_equal(initial['unit_ids'],ids):
        raise ValueError('privileged unit identity/order mismatch')
    dt=capture['dt_ms'];gap=initial['Theta_rel']-terminal['V_m_rel'];ref=terminal['refractory_step']
    if np.any(ref<0) or not np.array_equal(ref,np.rint(ref)):raise ValueError('native refractory ticks invalid')
    values=[];features=[]
    for name in names:
        mask=labels==name;g=gap[mask]
        pairs=[('voltage_gap_mean_mv',g.mean()),('voltage_gap_std_mv',g.std())]
        pairs += [(f'voltage_gap_q{q:g}_mv',np.quantile(g,q)) for q in config['voltage_gap_quantiles']]
        pairs += [(f'voltage_gap_le{v:g}mv_fraction',np.mean(g<=v)) for v in config['voltage_gap_thresholds_mv']]
        for key in ['I_syn_ex','I_syn_in']:
            state=terminal[key][mask];pairs += [(key+'_mean_pa',state.mean()),(key+'_std_pa',state.std())]
        pairs += [('refractory_fraction',np.mean(ref[mask]>0)),('refractory_mean_remaining_ms',np.mean(ref[mask])*dt)]
        values.extend(float(value) for _,value in pairs);features.extend(name+'_'+key for key,_ in pairs)
    result=np.asarray(values)
    if not np.isfinite(result).all():raise ValueError('nonfinite privileged summary')
    return result,np.asarray(features)


def build(acquisition, graph, boundaries, config_path, output):
    config=yaml.safe_load(config_path.read_text());design=json.loads((acquisition/'design.json').read_text())
    source=json.loads((graph/'manifest.json').read_text());graph_digest=sha256_file(graph/'manifest.json')
    if config['schema']!='neural_population_state_probe_v1' or config['cohort_role']!='development':
        raise ValueError('this diagnostic runner accepts only development cohorts')
    if graph_digest!=design['graph_manifest_sha256']:raise ValueError('acquisition source graph mismatch')
    if json.loads((boundaries/'acquisition.json').read_text())['status']!='ACQUIRED':raise ValueError('all declared boundary prefixes required')
    output.mkdir(parents=True,exist_ok=False)
    cases={(c['block'],c['focal_arrival_ms'],c['history'],c['present']):c for c in design['cases']}
    rows=[];rates=[];observed=[];privileged=[];target=[];source_rows=[];native_counts=set();phase_counts=set()
    names=source['resolved_source_network']['populations'][:-1];extent=source['resolved_source_network']['extent']
    delay=source['resolved_source_network']['th_delay_pulse_generator'];times=design['config']['focal_arrivals_ms']
    for block,timepoint,history in itertools.product(range(len(design['config']['initial_state_seeds'])),times,[False,True]):
        omitted=cases[(block,timepoint,history,False)];present=cases[(block,timepoint,history,True)]
        a,da,ia=_verified_events(acquisition/omitted['case_id'],omitted,graph_digest)
        b,db,ib=_verified_events(acquisition/present['case_id'],present,graph_digest)
        dt=a['dt_ms'];cutoff=int(ticks([timepoint-delay],dt)[0]);focal=int(ticks([timepoint],dt)[0]);width=int(ticks([design['config']['response_window_ms']],dt)[0])
        for key in ['unit_ids','populations','positions_mm']:
            if not np.array_equal(da[key],db[key]):raise ValueError('paired episode metadata mismatch')
        def before(d,i):return (d['spike_ticks']<cutoff)&(d['populations'][i]!='TC')
        pa,pb=before(da,ia),before(db,ib)
        if not all(np.array_equal(da[k][pa],db[k][pb]) for k in ['spike_ticks','spike_unit_ids']):raise ValueError('paired histories differ')
        def response(d,i):return np.sum((d['populations'][i]!='TC')&(d['spike_ticks']>=focal)&(d['spike_ticks']<focal+width))
        r,rn,s,sn=observed_features(da,ia,cutoff,dt,names,extent,config['features'])
        p,pn=privileged_features(boundaries/omitted['case_id'],da,names,config['features'])
        alignment=json.loads((boundaries/omitted['case_id']/'prefix_alignment.json').read_text())
        native_counts.add(int(np.sum(da['populations']!='TC')));phase_counts.add(alignment['sampled_phase_units'])
        if rows and (not np.array_equal(rn,rate_names) or not np.array_equal(sn,observed_names) or not np.array_equal(pn,privileged_names)):
            raise ValueError('feature definitions differ across rows')
        rate_names,observed_names,privileged_names=rn,sn,pn
        rates.append(r);observed.append(s);privileged.append(np.concatenate([s,p]));target.append((response(db,ib)-response(da,ia))/np.sum(da['populations']!='TC'))
        rows.append(dict(block=block,focal_arrival_ms=timepoint,history=int(history),cell=times.index(timepoint)*2+int(history),omitted_case=omitted['case_id']))
        source_rows.append(dict(omitted_report=sha256_file(acquisition/omitted['case_id']/'report.json'),present_report=sha256_file(acquisition/present['case_id']/'report.json'),boundary_capture=sha256_file(boundaries/omitted['case_id']/'boundary_capture.json'),alignment=sha256_file(boundaries/omitted['case_id']/'prefix_alignment.json')))
    np.savez_compressed(output/'representations.npz',R=np.array(rates),S=np.array(observed),P=np.array(privileged),R_names=rate_names,S_names=observed_names,
        P_names=np.concatenate([observed_names,privileged_names]),target=np.asarray(target),blocks=np.asarray([r['block'] for r in rows]),cells=np.asarray([r['cell'] for r in rows]))
    report=dict(status='BUILT_DEVELOPMENT',config=config,rows=rows,source_rows=source_rows,
        source_acquisition_sha256=sha256_file(acquisition/'design.json'),boundary_design_sha256=sha256_file(boundaries/'design.json'),
        arrays_sha256=sha256_file(output/'representations.npz'),feature_dimensions=dict(R=len(rate_names),S=len(observed_names),P=len(observed_names)+len(privileged_names)),
        full_hidden_state=False,privileged_state_phase_validation_units=sorted(phase_counts),privileged_native_units=sorted(native_counts),
        formal_training_completed=False,scientific_reference_qualified=False,implementation_sha256=sha256_file(Path(__file__)))
    (output/'features.json').write_text(json.dumps(report,indent=2)+'\n');return report



def analyze(dataset,output):
    source=json.loads((dataset/'features.json').read_text());config=source['config']
    if source['status']!='BUILT_DEVELOPMENT' or config['cohort_role']!='development':raise ValueError('complete development representations required')
    if sha256_file(dataset/'representations.npz')!=source['arrays_sha256']:raise ValueError('feature array changed')
    data=read_arrays(dataset/'representations.npz');blocks=data['blocks'];cells=data['cells'];y=data['target'];unique=np.unique(blocks)
    if len(unique)<4:raise ValueError('nested whole-block validation requires at least four blocks')
    output.mkdir(parents=True,exist_ok=False)
    models=[('cells','R','cells')]+[(rep+'_'+kind,rep,kind) for rep in ['R','S','P'] for kind in ['linear','rbf']]
    predictions=np.empty((len(y),len(models)));records=[];summaries=[]
    for j,(name,rep,kind) in enumerate(models):
        x=data[rep];grid=[1.] if kind=='cells' else config['predictors'][kind+'_regularization']
        for block in unique:
            train=blocks!=block;test=~train;scores=[]
            for alpha in grid:
                losses=[]
                for inner_block in unique[unique!=block]:
                    inner=train&(blocks!=inner_block);valid=blocks==inner_block
                    pred,_=conditional_prediction(x[inner],y[inner],cells[inner],x[valid],cells[valid],kind,alpha)
                    losses.append(float(np.mean((pred-y[valid])**2)))
                scores.append(float(np.mean(losses)))
            selected=grid[int(np.argmin(scores))]
            pred,diagnostic=conditional_prediction(x[train],y[train],cells[train],x[test],cells[test],kind,selected)
            fitted,_=conditional_prediction(x[train],y[train],cells[train],x[train],cells[train],kind,selected)
            predictions[test,j]=pred
            records.append(dict(model=name,block=int(block),selected_alpha=selected,inner_scores=json.dumps(scores),
                training_mse=float(np.mean((fitted-y[train])**2)),test_mse=float(np.mean((pred-y[test])**2)),**diagnostic))
        summary=dict(model=name,cv_mse=float(np.mean((predictions[:,j]-y)**2)))
        for t,timepoint in enumerate(sorted({r['focal_arrival_ms'] for r in source['rows']})):
            take=cells//2==t;summary[f'mse_at_{timepoint:g}ms']=float(np.mean((predictions[take,j]-y[take])**2))
        summaries.append(summary)
    for filename,rows in [('prediction_by_block.csv',records),('model_summary.csv',summaries)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    prediction_rows=[]
    for i,row in enumerate(source['rows']):prediction_rows.append(dict(row,target=float(y[i]),**{name:float(predictions[i,j]) for j,(name,_,_) in enumerate(models)}))
    with (output/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    np.savez_compressed(output/'predictions.npz',predictions=predictions,model_names=np.array([m[0] for m in models]),target=y,blocks=blocks,cells=cells)
    result=dict(status='ANALYZED_DEVELOPMENT',independent_blocks=len(unique),paired_response_rows=len(y),models=summaries,
        feature_dimensions=source['feature_dimensions'],feature_arrays_sha256=source['arrays_sha256'],predictions_sha256=sha256_file(output/'predictions.npz'),
        interpretation='descriptive nested CV; same six development blocks; no independent improvement interval or significance',
        full_hidden_state=False,formal_training_completed=False,neuroscience_finding_qualified=False,implementation_sha256=sha256_file(Path(__file__)))
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('build')
    for name in ['acquisition','graph','boundaries','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('analyze');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=build(a.acquisition,a.graph,a.boundaries,a.config,a.output) if a.action=='build' else analyze(a.dataset,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ['rows','source_rows']},indent=2))


if __name__=='__main__':main()

"""Causal terminal-state diagnostics from independent continuous pulse prefixes.

Native records precede arrival accumulation; terminal readback follows arrivals
at the last recorded tick. No unsupported continuation or state restoration is
used. Propagators and a full-run sampled prefix establish the observation phase.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from .nest_gpu_network import STATE_FIELDS, ticks
from .nest_gpu_spatial import Execution, run
from .provenance import sha256_file


def load_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def capture(graph, config, output):
    settings=Execution(**yaml.safe_load(config.read_text()))
    report=run(graph,output,settings)
    if report['status']!='EXECUTED':return 1
    ng=importlib.import_module('nestgpu')
    initial=load_arrays(output/'initial_parameters.npz');mapping=load_arrays(output/'unit_mapping.npz')
    native={int(u):int(v) for u,v in zip(mapping['source_ids'],mapping['native_ids'])}
    ids=initial['unit_ids'];node_ids=[native[int(u)] for u in ids]
    propagators={k:np.asarray(ng.GetNeuronStatus(node_ids,k)).reshape(-1) for k in ['P11ex','P11in']}
    if any(v.shape!=ids.shape or not np.isfinite(v).all() or np.any(v<np.finfo(np.float32).tiny) or np.any(v>1) for v in propagators.values()):
        raise ValueError('finite normal native current propagators required')
    np.savez_compressed(output/'terminal_propagators.npz',unit_ids=ids,**propagators)
    terminal=load_arrays(output/'terminal_states.npz')
    if not np.array_equal(terminal['unit_ids'],ids):raise ValueError('terminal and propagator unit IDs differ')
    receipt=dict(status='CAPTURED',source_report_sha256=sha256_file(output/'report.json'),
        terminal_states_sha256=sha256_file(output/'terminal_states.npz'),
        propagators_sha256=sha256_file(output/'terminal_propagators.npz'),
        native_phase='post-arrival after final recorded neuronal update',
        terminal_tick=int(ticks([settings.duration_ms],report['dt_ms'])[0]),dt_ms=report['dt_ms'],
        cortical_units=len(ids),full_hidden_state=False,scientific_reference_qualified=False,
        implementation_sha256=sha256_file(Path(__file__)))
    (output/'boundary_capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return 0


def verify(original, prefix, expected_end, emission, graph_digest):
    a=json.loads((original/'report.json').read_text());b=json.loads((prefix/'report.json').read_text())
    capture=json.loads((prefix/'boundary_capture.json').read_text())
    if (a['status']!='EXECUTED' or b['status']!='EXECUTED' or not all(b['checks'].values())
            or a['graph_manifest_sha256']!=graph_digest or b['graph_manifest_sha256']!=graph_digest
            or capture['source_report_sha256']!=sha256_file(prefix/'report.json')
            or capture['terminal_states_sha256']!=sha256_file(prefix/'terminal_states.npz')
            or capture['propagators_sha256']!=sha256_file(prefix/'terminal_propagators.npz')):
        raise ValueError('verified source/prefix reports required')
    for folder,report in [(original,a),(prefix,b)]:
        for name in ['spikes.npz','states.npz','initial_parameters.npz','spike_counts.npz']:
            if sha256_file(folder/name)!=report['artifacts'][name]:raise ValueError('prefix verification artifact changed')
    dt=b['dt_ms'];end=int(ticks([expected_end],dt)[0]);cutoff=int(ticks([emission],dt)[0])
    if not end<cutoff or end!=capture['terminal_tick']:raise ValueError('terminal state is not strictly causal')
    initial_a=load_arrays(original/'initial_parameters.npz');initial_b=load_arrays(prefix/'initial_parameters.npz')
    spikes_a=load_arrays(original/'spikes.npz');spikes_b=load_arrays(prefix/'spikes.npz')
    records_a=load_arrays(original/'states.npz');records_b=load_arrays(prefix/'states.npz')
    terminal=load_arrays(prefix/'terminal_states.npz');prop=load_arrays(prefix/'terminal_propagators.npz')
    take=spikes_a['spike_ticks']<=end
    checks=dict(initial_parameters_exact=initial_a.keys()==initial_b.keys() and all(np.array_equal(initial_a[k],initial_b[k]) for k in initial_a),
        all_unit_spike_prefix_exact=all(np.array_equal(spikes_a[k][take],spikes_b[k]) for k in ['spike_ticks','spike_unit_ids','raw_spike_times_ms']),
        metadata_exact=all(np.array_equal(spikes_a[k],spikes_b[k]) for k in ['unit_ids','populations','positions_mm']),
        state_ids_exact=np.array_equal(records_a['unit_ids'],records_b['unit_ids']),
        sampled_state_prefix_exact=np.array_equal(records_a['states'][records_a['ticks']<=end],records_b['states']),
        state_clock_exact=np.array_equal(records_b['ticks'],np.arange(end+1)),
        terminal_ids_exact=np.array_equal(terminal['unit_ids'],initial_b['unit_ids']) and np.array_equal(prop['unit_ids'],terminal['unit_ids']))
    select=np.searchsorted(terminal['unit_ids'],records_b['unit_ids'])
    if not np.array_equal(terminal['unit_ids'][select],records_b['unit_ids']):raise ValueError('unknown sampled state unit')
    fields=list(records_a['fields']);phase_errors={}
    for field in ['V_m_rel','refractory_step']:
        checks[field+'_terminal_equals_last_record']=np.array_equal(terminal[field][select],records_a['states'][end,:,fields.index(field)])
    for field,decay in [('I_syn_ex','P11ex'),('I_syn_in','P11in')]:
        expected=(terminal[field][select].astype(np.float32)*prop[decay][select].astype(np.float32)).astype(float)
        error=float(np.max(np.abs(expected-records_a['states'][end+1,:,fields.index(field)])))
        phase_errors[field]=error;checks[field+'_next_prearrival_record']=error<=.005
    result=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks,terminal_tick=end,emission_tick=cutoff,
        terminal_time_ms=end*dt,native_phase=capture['native_phase'],sampled_phase_units=len(select),
        full_cortical_terminal_units=len(terminal['unit_ids']),sampled_next_record_psc_error_pa=phase_errors,
        validation_only_next_tick=end+1,next_tick_is_predictor_input=False,
        full_hidden_state=False,scientific_reference_qualified=False)
    (prefix/'prefix_alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def acquire(acquisition, graph, library, output, resume=False):
    design=json.loads((acquisition/'design.json').read_text());manifest=json.loads((graph/'manifest.json').read_text())
    graph_digest=sha256_file(graph/'manifest.json')
    if graph_digest!=design['graph_manifest_sha256'] or sha256_file(library)!=design['source_rng_sha256']:
        raise ValueError('original graph and initial-state RNG required')
    dt=manifest['source_config']['simulation']['sim_resolution'];delay=manifest['resolved_source_network']['th_delay_pulse_generator']
    cases=[]
    for case in sorted(design['cases'],key=lambda c:(c['block'],c['focal_arrival_ms'],c['history'])):
        if case['present']:continue
        emission=case['focal_arrival_ms']-delay;end=(int(ticks([emission],dt)[0])-1)*dt
        settings=dict(case['settings'],duration_ms=end,source_rng_library=str(library.resolve()))
        Execution(**settings).validate()
        cases.append(dict(case_id=case['case_id'],emission_ms=emission,end_ms=end,settings=settings))
    declaration=dict(schema='mesocircuit_pulse_boundary_state_v1',source_acquisition_sha256=sha256_file(acquisition/'design.json'),
        graph_manifest_sha256=graph_digest,source_rng_sha256=sha256_file(library),cases=cases,
        native_phase='post-arrival at last tick, strictly before focal generator emission',
        comparison='full spike and sampled state prefixes; V/refractory last record; actual PSC decay to next record',
        expected_prefixes=len(cases),full_hidden_state=False,scientific_reference_qualified=False)
    if output.exists():
        if not resume or json.loads((output/'design.json').read_text())!=declaration:raise ValueError('resume requires identical prefix declaration')
    else:
        output.mkdir(parents=True);(output/'design.json').write_text(json.dumps(declaration,indent=2)+'\n')
    rows=[];started=time.perf_counter()
    for case in cases:
        folder=output/case['case_id'];config=output/(case['case_id']+'.yaml');text=yaml.safe_dump(case['settings'],sort_keys=False)
        if config.exists() and config.read_text()!=text:raise ValueError('prefix settings changed')
        config.write_text(text)
        cmd=[sys.executable,'-m','Libs.population_dynamics.pulse_boundary_state','capture','--graph',str(graph),'--config',str(config),'--output',str(folder)]
        reused=folder.exists()
        if reused:
            if not resume or asdict(Execution(**json.loads((folder/'report.json').read_text())['config']))!=asdict(Execution(**case['settings'])):raise ValueError('cannot reuse different prefix')
            code=0
        else:
            with (output/(case['case_id']+'.log')).open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL).returncode
        alignment=verify(acquisition/case['case_id'],folder,case['end_ms'],case['emission_ms'],graph_digest) if code==0 else {'status':'NOT_RUN'}
        rows.append(dict(case_id=case['case_id'],command=cmd,exit_code=code,reused=reused,alignment=alignment['status']))
        ledger=dict(status='RUNNING',expected_prefixes=len(cases),completed_prefixes=sum(x['exit_code']==0 and x['alignment']=='PASS' for x in rows),runs=rows,seconds=time.perf_counter()-started)
        if code or alignment['status']!='PASS':ledger['status']='FAILED'
        (output/'acquisition.json').write_text(json.dumps(ledger,indent=2)+'\n');print({k:v for k,v in ledger.items() if k!='runs'},flush=True)
        if ledger['status']=='FAILED':return 1
    ledger['status']='ACQUIRED';(output/'acquisition.json').write_text(json.dumps(ledger,indent=2)+'\n');return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    one=sub.add_parser('capture');one.add_argument('--graph',type=Path,required=True);one.add_argument('--config',type=Path,required=True);one.add_argument('--output',type=Path,required=True)
    many=sub.add_parser('acquire');many.add_argument('--acquisition',type=Path,required=True);many.add_argument('--graph',type=Path,required=True);many.add_argument('--source-rng-library',type=Path,required=True);many.add_argument('--output',type=Path,required=True);many.add_argument('--resume',action='store_true')
    a=p.parse_args();raise SystemExit(capture(a.graph,a.config,a.output) if a.action=='capture' else acquire(a.acquisition,a.graph,a.source_rng_library,a.output,a.resume))


if __name__=='__main__':main()

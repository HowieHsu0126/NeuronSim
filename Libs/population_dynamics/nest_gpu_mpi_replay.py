"""Independently replay source-capture and MPI regression evidence by unit ID."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import sha256_file


def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def verify(network, captures, output):
    checks={};details={}
    executions=json.loads((network/'verified_execution.json').read_text())
    checks['all_verified_commands_exited_zero']=all(r['exit_code']==0 for r in executions)
    drive=json.loads((network/'drive/drive.json').read_text())
    checks['original_effective_capture_identity']=sha256_file(captures/'effective/report.json')==drive['source_report_sha256']
    for schedule,dirname in [('effective','verified_distributed'),('initial','initial_distributed')]:
        original_report=json.loads((captures/schedule/'report.json').read_text())
        for filename in ['states.npz','spikes.npz','spike_counts.npz']:
            checks[f'{schedule}_original_{filename}_hash']=sha256_file(captures/schedule/filename)==original_report['artifacts'][filename]
        original=arrays(captures/schedule/'states.npz');events=arrays(captures/schedule/'spikes.npz')
        expected_units=set(map(int,original['unit_ids']));seen=set();seen_all=set();maxima=np.zeros(4);total_states=0
        index={int(u):i for i,u in enumerate(original['unit_ids'])}
        for rank in [0,1]:
            report=json.loads((network/dirname/f'rank_{rank}.json').read_text());data=arrays(network/dirname/f'rank_{rank}.npz')
            checks[f'{schedule}_rank_{rank}_arrays_hash']=sha256_file(network/dirname/f'rank_{rank}.npz')==report['arrays_sha256']
            checks[f'{schedule}_rank_{rank}_executed_and_counts']=report['status']=='EXECUTED' and report['checks']['native_incoming_connection_count_exact']
            checks[f'{schedule}_rank_{rank}_state_clock']=np.array_equal(data['ticks'],original['ticks']) and np.array_equal(data['fields'],original['fields'])
            for j,uid in enumerate(data['unit_ids']):
                uid=int(uid)
                if uid in seen or uid not in index:raise ValueError('duplicate or unknown cortical unit')
                seen.add(uid);delta=np.abs(data['states'][:,j]-original['states'][:,index[uid]])
                maxima=np.maximum(maxima,delta.max(axis=0));total_states+=delta.size
            for j,uid in enumerate(data['recorded_unit_ids']):
                uid=int(uid)
                if uid in seen_all:raise ValueError('duplicate recorded source identity')
                seen_all.add(uid);actual=np.sort(data['event_ticks'][data['event_ids']==uid]);wanted=np.sort(events['spike_ticks'][events['spike_unit_ids']==uid])
                checks[f'{schedule}_unit_{uid}_spikes_and_counter']=np.array_equal(actual,wanted) and len(actual)==int(data['native_counts'][j])
        checks[f'{schedule}_complete_cortex']=seen==expected_units
        checks[f'{schedule}_complete_all_recorded_units']=seen_all==set(map(int,events['unit_ids']))
        checks[f'{schedule}_states_within_existing_bounds']=bool(maxima[0]<=.0005 and max(maxima[1:3])<=.005 and maxima[3]==0)
        details[schedule]=dict(state_values=total_states,max_absolute_delta=dict(zip(original['fields'].tolist(),maxima.tolist())),source_spikes=len(events['spike_ticks']))
    for rank in [0,1]:
        old=arrays(network/'verified_distributed'/f'rank_{rank}.npz');fresh=arrays(network/'distributed_batch_replay'/f'rank_{rank}.npz')
        checks[f'rank_{rank}_fresh_process_changed_batch_arrays_exact']=old.keys()==fresh.keys() and all(np.array_equal(old[k],fresh[k]) for k in old)
        negative=arrays(network/'no_cross'/f'rank_{rank}.npz')
        checks[f'rank_{rank}_omission_changes_cortical_dynamics']=bool(np.any(old['states']!=negative['states']))
    routing=[]
    for primed in ['cold','primed']:
        root=network/'source_identity';reference=arrays(root/f'reference_{primed}'/'target_currents.npz')
        for mode in ['reference','bulk','unique']:
            d=root/f'{mode}_{primed}';source=json.loads((d/'rank_0.json').read_text());target=json.loads((d/f"rank_{0 if mode=='reference' else 1}.json").read_text());data=arrays(d/'target_currents.npz')
            checks[f'{mode}_{primed}_source_actual_ticks']=source['source_ticks']==[[10,30],[20,40]] and source['source_counters_exact']
            checks[f'{mode}_{primed}_raw_currents_hash']=sha256_file(d/'target_currents.npz')==target['target_arrays_sha256']
            expected=np.zeros_like(data['currents'])
            assignments=[0,1,0,1]+([0,1] if primed=='primed' else [None,None])
            for t in range(len(expected)-1):
                incoming=np.zeros(6)
                for neuron,origin in enumerate(assignments):
                    if origin is not None and t-10 in source['source_ticks'][origin]:incoming[neuron]=1024.
                expected[t+1]=(expected[t]+incoming)*data['propagator']
            error=float(np.max(np.abs(data['currents']-expected)))
            routing.append(dict(mode=mode,primed=primed,max_current_error_pa=error,routing_matches=error<=.005))
            if mode!='bulk':
                checks[f'{mode}_{primed}_independent_recurrence']=error<=.005
                checks[f'{mode}_{primed}_reference_currents_exact']=np.array_equal(data['currents'],reference['currents'])
    for mode,primed in [('reference',0),('unique',1)]:
        d=network/'source_identity'/f'{mode}_phase_{primed}'
        target=json.loads((d/f"rank_{0 if mode=='reference' else 1}.json").read_text())
        checks[f'{mode}_corrected_phase_cli']=target['status']=='EXECUTED' and target['routing_matches_declared_sources']
    result=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks,original_capture_comparisons=details,
        source_identity_regression=routing,
        retained_failure='bulk remote source mapping; original source-identity impulse oracle omitted within-step decay and is superseded by this independent recurrence',
        bounds=dict(voltage_mv=.0005,current_pa=.005,spike_ticks='exact'),
        native_poisson_qualified=False,scientific_reference_qualified=False,formal_training_completed=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--network',type=Path,required=True)
    p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.network,a.captures,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2));raise SystemExit(int(r['status']!='PASS'))


if __name__=='__main__':main()

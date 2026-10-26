"""Compare independent V1 native executions by source unit and exact event time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json

ACTIVITY_KEYS = {'spike_source_id', 'spike_time_ms', 'spike_counts', 'final_vm_mV'}
IDENTITY_KEYS = {'source_id', 'native_id', 'node_type_id', 'initial_vm_mV'}
INPUT_KEYS = {'population', 'source_id', 'native_id', 'source_event_counts'}


def compare_activity(first, second, units, duration_ms):
    for values in (first, second):
        if set(values) != ACTIVITY_KEYS:
            raise ValueError('V1_REPLAY_ACTIVITY_FIELDS_DIFFER')
        ids, times, counts, voltage = (values[k] for k in ('spike_source_id','spike_time_ms','spike_counts','final_vm_mV'))
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or times.shape != ids.shape
                or times.dtype.kind != 'f' or not np.isfinite(times).all()
                or np.any(ids < 0) or np.any(ids >= units) or np.any(times <= 0) or np.any(times > duration_ms)
                or counts.shape != (units,) or counts.dtype.kind not in 'iu'
                or voltage.shape != (units,) or voltage.dtype.kind != 'f' or not np.isfinite(voltage).all()
                or not np.array_equal(np.bincount(ids,minlength=units),counts)):
            raise ValueError('V1_REPLAY_ACTIVITY_IDENTITIES_OR_COUNTS_INVALID')
    checks = {}
    for key in ('spike_counts','final_vm_mV'):
        checks[key+'_exact'] = first[key].dtype == second[key].dtype and np.array_equal(first[key],second[key])
    first_order = np.lexsort((first['spike_time_ms'],first['spike_source_id']))
    second_order = np.lexsort((second['spike_time_ms'],second['spike_source_id']))
    checks['per_unit_spike_events_exact'] = all(
        first[key].dtype == second[key].dtype and np.array_equal(first[key][first_order],second[key][second_order])
        for key in ('spike_source_id','spike_time_ms'))
    return {'checks': {key:bool(value) for key,value in checks.items()},
            'compared_units':units, 'first_spikes':len(first_order), 'second_spikes':len(second_order),
            'global_recorder_row_order_equal':all(np.array_equal(first[key],second[key])
                for key in ('spike_source_id','spike_time_ms')),
            'comparison_rule':'Exact per-source-unit event multiset with multiplicity, exact counts and terminal voltage; no tolerance.'}


def load_outputs(directory, report):
    result = {}
    for name in ['cortical_identity.npz','cortical_activity.npz','input_identity.npz']:
        path = directory/name
        if sha256_file(path) != report['artifacts'][name]:
            raise ValueError('V1_REPLAY_OUTPUT_ARTIFACT_CHANGED:'+name)
        with np.load(path,allow_pickle=False) as archive:
            result[name] = {key:archive[key] for key in archive.files}
    if set(result['cortical_identity.npz']) != IDENTITY_KEYS or set(result['input_identity.npz']) != INPUT_KEYS:
        raise ValueError('V1_REPLAY_IDENTITY_FIELDS_DIFFER')
    return result


def compare_input_plans(first, second):
    """Compare raw generated inputs, including timestamps when counts agree."""
    keys = {'population', 'source_id', 'source_counts', 'source_event_unit', 'source_event_time_ms'}
    if set(first) != keys or set(second) != keys:
        raise ValueError('V1_REPLAY_GENERATED_INPUT_FIELDS_DIFFER')
    return {key: first[key].dtype == second[key].dtype and np.array_equal(first[key], second[key])
            for key in keys}


def compare(first_archive, second_working_directory, second_job_id, output):
    report = {'status':'FAILED_NATIVE_V1_REPEATABILITY','scientific_fidelity_qualified':False,
              'production_episodes':0,'started_utc':datetime.now(timezone.utc).isoformat(),
              'implementation_sha256':sha256_file(Path(__file__))}
    try:
        first, delivered = bound_json(first_archive,'network/report.json')
        second = json.loads((second_working_directory/'network/report.json').read_text())
        if str(delivered['job_id']) == str(second_job_id):
            raise ValueError('V1_REPLAY_REQUIRES_INDEPENDENT_EXECUTION')
        fields = ['implementation_sha256','nest_version','threads','connection_batch_size','kernel_rng_seed',
                  'coupled_cortical_neurons','native_duration_ms','input_generators','input_events',
                  'recurrent_connections','input_connections','native_total_connections','observation_scope']
        if (any(first[key] != second[key] for key in fields)
                or first['started_utc'] == second['started_utc']
                or any(row['status'] != 'NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE' for row in (first,second))
                or delivered['status'] != 'PASS' or delivered['exit_code'] != 0):
            raise ValueError('V1_REPLAY_EXECUTION_CONTRACT_DIFFERS')
        consumed = ['graph/graph_identity.json','reference/reference/intake.json','parameters/comparison.json',
                    'inputs/input_identity.json','pointnet/source_inspection.json','timing/diagnosis/report.json']
        for relative in consumed:
            if sha256_file(first_archive/relative) != sha256_file(second_working_directory/relative):
                raise ValueError('V1_REPLAY_SOURCE_EVIDENCE_DIFFERS:'+relative)
        a, b = load_outputs(first_archive/'network',first), load_outputs(second_working_directory/'network',second)
        checks = {}
        for name in ['cortical_identity.npz','input_identity.npz']:
            for key in a[name]:
                checks[name+':'+key] = a[name][key].dtype == b[name][key].dtype and np.array_equal(a[name][key],b[name][key])
        generated_metadata = [{k: v for k, v in row.get('generated_lgn', {}).items() if k != 'source_archive'}
                              for row in (first, second)]
        if generated_metadata[0] != generated_metadata[1]:
            raise ValueError('V1_REPLAY_GENERATED_INPUT_CONDITION_DIFFERS')
        if generated_metadata[0]:
            plans = []
            for directory, native in ((first_archive/'network', first), (second_working_directory/'network', second)):
                path = directory/'source_inputs.npz'
                if sha256_file(path) != native['artifacts']['source_inputs.npz']:
                    raise ValueError('V1_REPLAY_GENERATED_INPUT_ARTIFACT_CHANGED')
                with np.load(path, allow_pickle=False) as values:
                    plans.append({key: values[key] for key in values.files})
            checks.update({'source_inputs.npz:'+key: bool(value) for key, value in compare_input_plans(*plans).items()})
        activity = compare_activity(a['cortical_activity.npz'],b['cortical_activity.npz'],first['coupled_cortical_neurons'],first['native_duration_ms'])
        checks.update(activity['checks'])
        report.update(first_job_id=delivered['job_id'],second_job_id=str(second_job_id),
                      first_archive=str(first_archive),second_working_directory=str(second_working_directory),
                      first_report_sha256=sha256_file(first_archive/'network/report.json'),
                      second_report_sha256=sha256_file(second_working_directory/'network/report.json'),
                      consumed_source_evidence_identical=True,checks={key:bool(value) for key,value in checks.items()},
                      activity=activity,status='PASS_NATIVE_V1_REPEATABILITY' if all(checks.values()) else 'FAILED_NATIVE_V1_REPEATABILITY')
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        freeze_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['first-archive','second-working-directory','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--second-job-id',required=True)
    report=compare(**vars(parser.parse_args()))
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS_NATIVE_V1_REPEATABILITY' else 2


if __name__=='__main__':
    raise SystemExit(main())

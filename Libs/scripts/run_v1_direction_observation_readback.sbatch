#!/bin/bash
#SBATCH --job-name=NrnSim_V1ObservationReadback
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_observation_readback-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/conda/bin/python3 -B - "$1" "$2" "$3" "$0" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback

repo, submission_path = map(Path, sys.argv[1:3])
directions = [int(value) for value in sys.argv[3].split(',')]
job = os.environ['SLURM_JOB_ID']
assert os.environ['SLURMD_NODENAME'] == 'cpu04' and not os.environ.get('SLURM_JOB_GPUS')
assert directions and len(set(directions)) == len(directions)
submission = json.loads(submission_path.read_text())
source_submission = json.loads(Path(submission['source_submission']).read_text())
family = repo/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route'
construction = family/'circuit_construction_source'
output = construction/'direction_coverage/observation_readback'/('run_'+job)
output.mkdir(parents=True, exist_ok=False)
report = dict(status='PREPARING', job_id=job, directions_deg=directions,
              source_submission=str(submission_path), started_utc=datetime.now(timezone.utc).isoformat(),
              cases=[], new_source_simulations=0, new_model_generations=0, production_episodes=0,
              scientific_qualification=False, complete_eight_direction_readout=False)
code = 1

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def manifest(archive):
    value = json.loads((archive/'delivery.json').read_text())
    return value, {row['path']: row for row in value['files']}

def copy_checked(archive, entries, relative, destination):
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        raise ValueError('DELIVERY_RELATIVE_PATH_REQUIRED')
    entry = entries[relative]
    source = archive/relative
    if source.stat().st_size != entry['bytes']:
        raise ValueError('DELIVERY_SIZE_CHANGED:'+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != entry['sha256']:
        raise ValueError('DELIVERY_CONTENT_CHANGED:'+str(source))

try:
    validation = Path(submission['software_validation'])
    vd, vi = manifest(validation)
    if (vd['status'] != 'PASS' or vd['exit_code'] != 0
            or vd['mode'] != 'constructed-controls-validate' or not vd['validation_only']
            or vd['junit'] != dict(tests=88, failures=0, errors=0, skipped=0)):
        raise ValueError('REGISTERED_OBSERVATION_SOFTWARE_VALIDATION_REQUIRED')
    jobs = {(row['direction_deg'], row['phase']): row['job_id'] for row in submission['jobs']}
    native_jobs = {row['direction_deg']: row['job_id'] for row in source_submission['jobs']
                   if row['phase'] == 'full_native_and_replay'}
    if not set(directions).issubset(native_jobs):
        raise ValueError('UNREGISTERED_DIRECTION')
    reference = Path(submission['reused_315_observation'])
    rd, ri = manifest(reference)
    if rd['status'] != 'PASS' or rd['exit_code'] != 0:
        raise ValueError('VERIFIED_REFERENCE_OBSERVATION_REQUIRED')
    with tempfile.TemporaryDirectory(prefix='neuronsim-v1-observation-readback-'+job+'.', dir='/tmp') as temp:
        work = Path(temp)
        for relative in vi:
            if relative.startswith('Libs/'):
                copy_checked(validation, vi, relative, work/relative)
        sys.path.insert(0, str(work/'Libs'))
        import h5py
        import numpy as np
        from population_dynamics.resource_probe import probe
        from population_dynamics.circuit_data import CircuitDataset, CircuitEpisode
        from population_dynamics.v1_source_dataset import (
            read_pointnet_outputs, prepare_spikes, pointnet_input_evidence,
            pointnet_case_metadata, verify_export)
        from population_dynamics.v1_visual_controls import load_visual_controls
        admission = probe(work/'admission.json', required_memory_gib=4)
        shutil.copy2(work/'admission.json', output/'admission.json')
        if admission['status'] != 'PASS':
            raise RuntimeError('OBSERVATION_READBACK_RESOURCE_BLOCKED:'+str(admission['reasons']))
        for relative in ('export/report.json', 'export/dataset/dataset.json'):
            copy_checked(reference, ri, relative, work/'reference'/relative)
        reference_report = json.loads((work/'reference/export/report.json').read_text())
        circuit_id = reference_report['circuit_id']
        reference_catalog = json.loads((work/'reference/export/dataset/dataset.json').read_text())
        static = reference_catalog['circuits'][circuit_id]
        static_relative = 'export/dataset/'+static['path']
        copy_checked(reference, ri, static_relative, work/'reference'/static_relative)
        with h5py.File(work/'reference'/static_relative, 'r') as source:
            units = SimpleNamespace(circuit_id=circuit_id,
                **{name: source[name][:] for name in ('unit_ids', 'node_type_ids', 'coordinates', 'field_index')},
                populations=source['populations'].asstr()[:], field_names=tuple(source['field_names'].asstr()[:]))
        if len(units.unit_ids) != 230924:
            raise ValueError('COMPLETE_REFERENCE_CORTICAL_INVENTORY_REQUIRED')
        reference_native = Path(rd['pointnet_native_archive'])
        _, reference_native_index = manifest(reference_native)
        copy_checked(reference_native, reference_native_index, 'primary/report.json', work/'reference_native.json')
        reference_native_report = json.loads((work/'reference_native.json').read_text())
        for degree in directions:
            native_archive = construction/'native_pointnet'/('run_'+str(native_jobs[degree]))
            controls_archive = family/'source_observation'/('run_'+str(jobs[degree, 'Controls']))
            export_archive = family/'source_observation'/('run_'+str(jobs[degree, 'Observation']))
            nd, ni = manifest(native_archive)
            cd, ci = manifest(controls_archive)
            ed, ei = manifest(export_archive)
            if any(value.get('status') != 'PASS' or value.get('exit_code') != 0
                   or value.get('tests_reused') is not True
                   or value.get('validation_reference') != str(validation) for value in (cd, ed)):
                raise ValueError('PASSED_BOUND_CONTROL_AND_EXPORT_DELIVERIES_REQUIRED')
            if (ed['pointnet_native_archive'] != str(native_archive)
                    or ed['visual_controls_archive'] != str(controls_archive)):
                raise ValueError('REGISTERED_NATIVE_CONTROL_EXPORT_BINDING_DIFFERS')
            native_files = ['delivery.json', 'primary/report.json', 'replay/report.json', 'comparison/report.json']
            native_files += [f'{prefix}/native/{name}' for prefix in ('primary', 'replay')
                             for name in ('cortical_spikes.npz', 'cortical_state.npz', 'input_events.npz', 'parameters.json')]
            stage_bytes = sum(ni[name]['bytes'] for name in native_files if name != 'delivery.json')
            stage_bytes += sum(row['bytes'] for name, row in ei.items() if name.startswith('export/'))
            stage_bytes += ci['controls/commands.npz']['bytes']
            if shutil.disk_usage(work).free < stage_bytes+20*1024**3:
                raise ValueError('LOCAL_SCRATCH_RESERVE_REQUIRED')
            with tempfile.TemporaryDirectory(prefix=f'direction-{degree:03d}-', dir=work) as case_temp:
                case = Path(case_temp)
                for archive, entries, target, names in (
                    (native_archive, ni, case/'native', native_files),
                    (controls_archive, ci, case/'controls', ['delivery.json', 'controls/report.json', 'controls/commands.npz']),
                    (export_archive, ei, case/'observation', [name for name in ei if name.startswith('export/')])):
                    for relative in names:
                        if relative == 'delivery.json':
                            target.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(archive/relative, target/relative)
                        else:
                            copy_checked(archive, entries, relative, target/relative)
                native, _, replay, arrays, _ = read_pointnet_outputs(case/'native')
                if (native['cortical_units'] != 230924 or native['mode'] != 'full'
                        or native['bindings']['spike_inputs']['config']['stimulus']['direction_deg'] != degree):
                    raise ValueError('REGISTERED_FULL_DIRECTION_REQUIRED')
                for key in ('recurrent', 'inputs', 'parameters', 'point_config', 'sources'):
                    if native['bindings'][key] != reference_native_report['bindings'][key]:
                        raise ValueError('REFERENCE_CIRCUIT_BINDING_CHANGED:'+key)
                clock = native['bindings']['point_config']['source_clock']
                if clock != dict(duration_ms=3000., resolution_ms=.25):
                    raise ValueError('REGISTERED_SOURCE_CLOCK_CHANGED')
                state, spikes = arrays['cortical_state.npz'], arrays['cortical_spikes.npz']
                if not np.array_equal(state['source_id'], units.unit_ids):
                    raise ValueError('REFERENCE_UNIT_IDENTITIES_CHANGED')
                ticks, indices = prepare_spikes(dict(spike_time_ms=spikes['time_ms'],
                    spike_source_id=spikes['source_id'], spike_counts=state['spike_count']), units, 12000, .25)
                input_identity, privileged = pointnet_input_evidence(native, arrays['input_events.npz'])
                endpoint = ticks == 12000
                privileged.update(initial_vm_mV=state['initial_vm_mV'], terminal_vm_mV=state['terminal_vm_mV'],
                    final_endpoint_unit_indices=indices[endpoint], final_endpoint_ticks=ticks[endpoint])
                controls = load_visual_controls(case/'controls', native, native_family='pointnet')
                episode_id, block_id, seeds, _ = pointnet_case_metadata(native, circuit_id,
                    state['initial_vm_mV'], input_identity, clock)
                episode = CircuitEpisode(episode_id=episode_id, circuit_id=circuit_id, block_id=block_id,
                    split='development', duration_ticks=12000, dt_ms=.25, spike_ticks=ticks,
                    spike_unit_indices=indices, privileged=privileged, seeds=seeds,
                    source_record=dict(complete_unit_recording=True, simulation_completed=True,
                        recorded_until_tick=12000, expected_duration_ticks=12000, core_unit_count=230924),
                    **{key: controls[key] for key in ('control_ticks', 'control_values', 'control_features', 'control_kind')})
                episode.validate(230924)
                dataset = CircuitDataset(case/'observation/export/dataset', splits=('development',))
                if set(dataset.episodes) != {episode_id} or dataset.episodes[episode_id]['block_id'] != block_id:
                    raise ValueError('EXPORTED_EPISODE_IDENTITY_DIFFERS')
                actual = verify_export(dataset, episode, units)
                declared = json.loads((case/'observation/export/report.json').read_text())
                if (actual != declared['verification'] or declared['episode_id'] != episode_id
                        or declared['native_job_id'] != str(native_jobs[degree])):
                    raise ValueError('EXPORTED_VERIFICATION_DIFFERS_FROM_NATIVE_READBACK')
                report['cases'].append(dict(direction_deg=degree, native_job=native_jobs[degree],
                    controls_job=jobs[degree, 'Controls'], observation_job=jobs[degree, 'Observation'],
                    native_delivery_sha256=digest(native_archive/'delivery.json'),
                    controls_delivery_sha256=digest(controls_archive/'delivery.json'),
                    observation_delivery_sha256=digest(export_archive/'delivery.json'),
                    native_primary_report_sha256=digest(case/'native/primary/report.json'),
                    export_report_sha256=digest(case/'observation/export/report.json'),
                    episode_id=episode_id, block_id=block_id, circuit_id=circuit_id,
                    primary_replay_equal=replay['all_native_outputs_equal'], verification=actual))
        if len({row['episode_id'] for row in report['cases']}) != len(directions):
            raise ValueError('DUPLICATE_DIRECTION_EPISODE_ID')
        report.update(status='PASS_BOUND_DIRECTION_SOURCE_OBSERVATION_READBACK',
            independent_circuit_realizations=1, tests_reused=vd['junit'],
            validation_delivery_sha256=digest(validation/'delivery.json'),
            reference_observation_delivery_sha256=digest(reference/'delivery.json'))
        code = 0
except Exception as error:
    report.update(status='FAILED_DIRECTION_OBSERVATION_READBACK', error_type=type(error).__name__,
                  error=str(error), traceback=traceback.format_exc())
finally:
    report.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(sys.argv[4], output/'launcher.sbatch')
    files = [dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
             for p in (output/'report.json', output/'launcher.sbatch', output/'admission.json') if p.exists()]
    (output/'delivery.json').write_text(json.dumps(dict(status=report['status'], job_id=job,
        exit_code=code, files=files), indent=2)+'\n')
    print(json.dumps(report), flush=True)
raise SystemExit(code)
PY

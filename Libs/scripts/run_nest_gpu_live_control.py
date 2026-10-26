"""Build, qualify and execute source-bound native future pulse control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import yaml

from run_population_trajectory_campaign import COHORT, archive, copy_bound, manifest_rows

GRAPH = 'Output/neural_population_dynamics/mechanistic_reference/spatial_connectivity/source_graph_100000'
COUPLED_CASE = 'train_block_014_familiar_0'
FUTURE_ENSEMBLE_CONFIG = 'Libs/configs/neural_population_native_future_ensemble.yaml'
MATCHED_FORECAST_CONFIG = 'Libs/configs/neural_population_matched_forecast.yaml'
ACCEPTED_SEED41 = 'Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_41'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(repo, work, target, future_poisson_rng=False):
    if os.environ.get('SLURMD_NODENAME') != 'cpu04' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('CPU04 Slurm build required')
    target.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', job_id=os.environ['SLURM_JOB_ID'], work=str(work),
                  started_utc=datetime.now(timezone.utc).isoformat(), commands=[],
                  scientific_claim_qualified=False, gpu_simulation_performed=False)
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PATH='/usr/bin:/usr/local/cuda/bin:/bin:' + os.environ.get('PATH', ''), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TMPDIR=str(work/'tmp'), PYTHONDONTWRITEBYTECODE='1')
    (work/'tmp').mkdir()
    def save():
        (target/'build.json').write_text(json.dumps(record, indent=2)+'\n')
    def command(name, args):
        with (work/(name+'.log')).open('w') as stream:
            result = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
        shutil.copyfile(work/(name+'.log'), target/(name+'.log'))
        record['commands'].append(dict(name=name, args=args, exit_code=result.returncode))
        save()
        if result.returncode:
            raise RuntimeError(name+' failed; inspect '+str(target/(name+'.log')))
    try:
        source = repo/'Output/neural_population_dynamics/mechanistic_reference/nest_gpu_archive_source/nest-gpu.tar.gz'
        archive = work/'source.tar.gz'
        shutil.copyfile(source, archive)
        if digest(archive) != '63076b1c040bdeaaedbb482d354fb46b42fdc300401a1c7c7309e0b491d78549':
            raise ValueError('pinned source archive mismatch')
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            prefix = members[0].name.split('/')[0]
            stream.extractall(work/'unpacked')
        source = work/'source'
        (work/'unpacked'/prefix).rename(source)
        record['source_revision'] = '74fc021cad2b4bd5c011c390b3f95a302b663483'
        patches = [repo/'Libs/external/nest_gpu/patches/no_mpi_accessor.patch',
                   repo/'Output/neural_population_dynamics/mechanistic_reference/nest_gpu_poisson_production/native_rng_variants/philox_variant_preparation/per_edge_philox.patch',
                   repo/'Libs/external/nest_gpu/patches/live_spike_schedule.patch']
        if future_poisson_rng:
            patches.append(repo/'Libs/external/nest_gpu/patches/future_poisson_rng.patch')
        record['future_poisson_rng_interface_built'] = future_poisson_rng
        record['patches'] = []
        for index, patch in enumerate(patches):
            local = work/patch.name
            shutil.copyfile(patch, local)
            record['patches'].append(dict(path=str(patch.relative_to(repo)), sha256=digest(local)))
            shutil.copyfile(local, target/local.name)
            command('patch_'+str(index), ['patch', '--batch', '--forward', '-p1', '--directory='+str(source), '--input='+str(local)])
        record['toolchain'] = {name:subprocess.check_output(args, text=True, env=env)
                               for name,args in [('cuda',['/usr/local/cuda/bin/nvcc','--version']),
                                                 ('cxx',['/usr/bin/c++','--version']),('cmake',['cmake','--version'])]}
        command('configure', ['cmake', '-S', str(source), '-B', str(work/'build'), '-Dwith-mpi=OFF',
                             '-Dwith-gpu-arch=89', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
                             '-DCMAKE_CXX_COMPILER=/usr/bin/c++', '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++', '-DCMAKE_INSTALL_PREFIX='+str(work/'install'),
                             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS=-DNEURONSIM_POISSON_PHILOX=1',
                             '-DCMAKE_CUDA_FLAGS=-DNEURONSIM_POISSON_PHILOX=1', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
        command('compile', ['cmake', '--build', str(work/'build'), '--parallel', '2'])
        command('install', ['cmake', '--install', str(work/'build')])
        library = work/'install/lib64/nestgpu/libnestgpukernel.so'
        shutil.copyfile(library, target/library.name)
        shutil.copyfile(source/'pythonlib/nestgpu.py', target/'nestgpu.py')
        record.update(status='BUILT_NATIVE_CANARY_PENDING', library_sha256=digest(library),
                      library=str(target/library.name))
    except Exception as error:
        record.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return record


def check(repo, work, target, native_build, coupled_replay=False, prepare_only=False, canary=None, canary_software=None, producer_software=None, future_poisson_rng_canary=False):
    if coupled_replay:
        return replay_coupled(repo, work, target, native_build, prepare_only, canary, producer_software)
    if (os.environ.get('SLURMD_NODENAME') != 'gpu03' or not os.environ.get('SLURM_JOB_GPUS')
            or not os.environ.get('SLURM_JOB_ID')):
        raise ValueError('GPU03 Slurm GPU allocation required')
    target.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    record = dict(status='PREPARING', job_id=os.environ['SLURM_JOB_ID'], work=str(work),
                  started_utc=datetime.now(timezone.utc).isoformat(), scientific_claim_qualified=False)
    code = 2
    try:
        proof = json.loads((native_build/'build.json').read_text())
        if proof['status'] != 'BUILT_NATIVE_CANARY_PENDING':
            raise ValueError('completed isolated native build required')
        if future_poisson_rng_canary and proof.get('future_poisson_rng_interface_built') is not True:
            raise ValueError('future Poisson RNG interface build required')
        for name in ['libnestgpukernel.so', 'nestgpu.py', 'build.json']:
            shutil.copyfile(native_build/name, work/name)
        if digest(work/'libnestgpukernel.so') != proof['library_sha256']:
            raise ValueError('native build library binding mismatch')
        original = (repo/'Output/neural_population_dynamics/model/native_live_control_build/run_606594/libnestgpukernel.so'
                    if future_poisson_rng_canary else
                    repo/'Output/neural_population_dynamics/mechanistic_reference/nest_gpu_poisson_production/native_rng_variants/libraries/philox/libnestgpukernel.so')
        shutil.copyfile(original, work/'original_libnestgpukernel.so')
        original_digest = ('78c8d420759064ccc209769e1ea04f48a01a3f463c0cb9d908c0e1a4c85eea6c'
                           if future_poisson_rng_canary else
                           '1fade400620f7498d4dba13b4753e09f79391f55770cd0e426e7fead8f87c0e6')
        if digest(work/'original_libnestgpukernel.so') != original_digest:
            raise ValueError('original Philox library binding mismatch')
        files = ['nest_gpu_live_control.py', 'nest_gpu_live_control_check.py',
                 'mesocircuit_continuation.py', 'resource_probe.py']
        record['implementation'] = {}
        software = canary_software or repo/'Output/neural_population_dynamics/model/native_live_control_software/run_606766'
        tested = json.loads((software/'delivery.json').read_text())
        if tested['status'] != 'PASS' or tested['exit_code'] != 0:
            raise ValueError('accepted native control software required')
        rows = manifest_rows(tested)
        record['canary_software'] = str(software)
        record['canary_software_delivery_sha256'] = digest(software/'delivery.json')
        for name in files:
            relative = 'Libs/population_dynamics/'+name
            local = work/relative
            local.parent.mkdir(parents=True, exist_ok=True)
            if relative in rows:
                copy_bound(software/relative, local, rows[relative])
            else:
                shutil.copyfile(repo/relative, local)
            record['implementation'][name] = digest(local)
        (work/'Libs/__init__.py').touch()
        (work/'Libs/population_dynamics/__init__.py').touch()
        environment = dict(os.environ, NESTGPU_LIB=str(work/'libnestgpukernel.so'),
                           PYTHONPATH=str(work), PYTHONDONTWRITEBYTECODE='1',
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (work/'resource.log').open('w') as stream:
            result = subprocess.run(['python3', '-m', 'Libs.population_dynamics.resource_probe',
                                     '--output', str(work/'resource.json'), '--required-memory-gib', '2'],
                                    cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('allocated-node resource check failed')
        command = ['python3', '-m', 'Libs.population_dynamics.nest_gpu_live_control_check',
                   '--output', str(work/'results'), '--original-library', str(work/'original_libnestgpukernel.so')]
        if future_poisson_rng_canary:
            command.append('--future-rng-suite')
        record.update(status='RUNNING', command=command, native_library_sha256=proof['library_sha256'])
        (target/'delivery.json').write_text(json.dumps(record, indent=2)+'\n')
        with (work/'execution.log').open('w') as stream:
            result = subprocess.run(command, cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        code = result.returncode
        report = json.loads((work/'results/report.json').read_text())
        if code == 0 and report['status'] != 'PASS':
            raise ValueError('native canary exit status contradicts report')
        record.update(status='PASS' if code == 0 else 'FAILED', native_canary_status=report['status'],
                      future_poisson_rng_canary=future_poisson_rng_canary)
    except Exception as error:
        code = code or 2
        record.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
    finally:
        for name in ['resource.json', 'resource.log', 'execution.log']:
            if (work/name).exists():
                shutil.copyfile(work/name, target/name)
        if (work/'results').exists():
            shutil.copytree(work/'results', target/'results')
        if (work/'Libs').exists():
            shutil.copytree(work/'Libs', target/'Libs')
        record.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat())
        (target/'delivery.json').write_text(json.dumps(record, indent=2)+'\n')
    return code


def stage_coupled_inputs(repo, work, native_build, producer_software=None):
    """Bind one complete development episode without reading graph payloads."""
    source = repo/COHORT
    delivery = json.loads((source/'delivery.json').read_text())
    if (delivery['status'] != 'VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE'
            or delivery['completed_episodes'] != 72 or delivery['cortical_units'] != 100000):
        raise ValueError('complete source cohort required')
    index = manifest_rows(delivery)
    copy_bound(source/'train/design.json', work/'inputs/design.json', index['train/design.json'])
    design = json.loads((work/'inputs/design.json').read_text())
    case = next(c for c in design['cases'] if c['case_id'] == COUPLED_CASE)
    if (case['split'] != 'train' or case['block'] != 14 or case['initial_state_seed'] != 83015
            or case['background_seed'] != 84015):
        raise ValueError('fixed development replay identity required')
    copy_bound(source/'dependencies/source_rng.so', work/'source_rng.so', index['dependencies/source_rng.so'])
    raw = 'train/'+COUPLED_CASE+'/'
    copy_bound(source/(raw+'report.json'), work/'reference/report.json', index[raw+'report.json'])
    reference = json.loads((work/'reference/report.json').read_text())
    if reference['status'] != 'EXECUTED' or reference['duration_ms'] != 6000 or reference['cortical_neurons'] != 100000:
        raise ValueError('accepted full native episode required')
    arrays = ['initial_parameters.npz', 'spike_counts.npz', 'spikes.npz', 'states.npz', 'terminal_states.npz']
    arrays += [f'state_observation_{tick:09d}.npz' for tick in case['settings']['state_observation_ticks']]
    for name in arrays:
        copy_bound(source/(raw+name), work/'reference'/name, index[raw+name])
    graph_source = repo/GRAPH
    manifest_path = work/'graph_manifest.json'
    shutil.copyfile(graph_source/'manifest.json', manifest_path)
    if digest(manifest_path) != reference['graph_manifest_sha256']:
        raise ValueError('graph identity differs from accepted episode')
    manifest = json.loads(manifest_path.read_text())
    if manifest['cortical_neurons'] != 100000 or manifest['edges'] != 393052910:
        raise ValueError('full coupled source graph required')
    if digest(work/'source_rng.so') != manifest['library_sha256']:
        raise ValueError('original initial-state RNG identity changed')
    # This boundary commits the third source pulse while its TC arrival is pending.
    # Installing the identical effective schedule tests consumed indices and queues.
    settings = dict(case['settings'])
    initial = settings['pulse_control']['initial_emission_ms']
    boundary = initial[2]
    settings.update(source_rng_library=str(work/'source_rng.so'), live_pulse_control=True,
                    pulse_control=dict(initial_emission_ms=initial, decision_time_ms=boundary,
                                       future_emission_ms=[t for t in initial if t > boundary]))
    (work/'execution.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    software = producer_software or repo/'Output/neural_population_dynamics/model/native_live_control_software/run_606766'
    proof = json.loads((software/'delivery.json').read_text())
    if proof['status'] != 'PASS' or proof['exit_code'] != 0:
        raise ValueError('accepted source identity and producer software required')
    rows = manifest_rows(proof)
    for name, row in rows.items():
        if name.endswith('.py'):
            copy_bound(software/name, work/name, row)
    # The full replay invokes this module before staging the graph. The
    # producer software receipt does not include the resource probe.
    resource_source = repo/'Output/neural_population_dynamics/model/trajectory_software/run_606023'
    resource_proof = json.loads((resource_source/'delivery.json').read_text())
    if resource_proof['status'] != 'PASS' or resource_proof['exit_code'] != 0:
        raise ValueError('accepted allocation resource probe required')
    resource_rows = manifest_rows(resource_proof)
    resource_name = 'Libs/population_dynamics/resource_probe.py'
    copy_bound(resource_source/resource_name, work/resource_name, resource_rows[resource_name])
    (work/'Libs/__init__.py').touch()
    native = json.loads((native_build/'build.json').read_text())
    if native['status'] != 'BUILT_NATIVE_CANARY_PENDING':
        raise ValueError('accepted native build required')
    for name in ['libnestgpukernel.so', 'nestgpu.py']:
        shutil.copyfile(native_build/name, work/name)
    if digest(work/'libnestgpukernel.so') != native['library_sha256']:
        raise ValueError('native replay library binding mismatch')
    record = dict(case_id=COUPLED_CASE, graph_manifest_sha256=digest(manifest_path),
                  original_report_sha256=digest(work/'reference/report.json'),
                  native_library_sha256=native['library_sha256'],
                  source_rng_sha256=digest(work/'source_rng.so'), settings=settings,
                  core_neurons=100000, duration_ms=6000, decision_time_ms=boundary,
                  pending_committed_tc_arrival_ms=boundary+manifest['resolved_source_network']['th_delay_pulse_generator'],
                  graph_parts=len(manifest['parts']), graph_bytes=sum((graph_source/row['path']).stat().st_size for row in manifest['parts']),
                  producer_software=str(software), producer_delivery_sha256=digest(software/'delivery.json'),
                  reference_arrays=arrays, graph_payload_staged=False, mechanistic_simulation_performed=False,
                  reason='validate changed live interface on a full coupled episode; same effective control as accepted reference',
                  scientific_claim_qualified=False)
    (work/'plan.json').write_text(json.dumps(record, indent=2)+'\n')
    return record, manifest


def compare_coupled_replay(work, expected_names):
    comparisons = []
    for name in expected_names:
        with np.load(work/'reference'/name, allow_pickle=False) as original, np.load(work/'results'/name, allow_pickle=False) as actual:
            if set(original.files) != set(actual.files):
                raise ValueError('native replay array schema changed: '+name)
            for field in original.files:
                left, right = original[field], actual[field]
                exact = left.dtype == right.dtype and np.array_equal(left, right)
                comparisons.append(dict(artifact=name, array=field, shape=list(left.shape), values=int(left.size), exact=exact))
    report = dict(status='PASS' if all(r['exact'] for r in comparisons) else 'FAILED',
                  comparisons=comparisons, compared_values=sum(r['values'] for r in comparisons),
                  scientific_claim_qualified=False, original_effective_schedule_unchanged=True)
    (work/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def stage_native_graph(repo, work, graph, manifest):
    """Restore only manifest-bound graph payloads on the allocated GPU node."""
    graph.mkdir(exist_ok=False)
    shutil.copyfile(work/'graph_manifest.json', graph/'manifest.json')
    layout = repo/GRAPH/'layout.npz'
    copy_bound(layout, graph/'layout.npz', dict(bytes=layout.stat().st_size, sha256=manifest['layout_sha256']))
    for row in manifest['parts']:
        source = repo/GRAPH/row['path']
        copy_bound(source, graph/row['path'], dict(bytes=source.stat().st_size, sha256=row['sha256']))


def check_coupled_future_noise(repo, work, target, native_build, canary, producer_software):
    """Qualify one uniform-delay stream switch on the existing full source."""
    if (os.environ.get('SLURMD_NODENAME') != 'gpu03' or not os.environ.get('SLURM_JOB_GPUS')
            or not os.environ.get('SLURM_JOB_ID')):
        raise ValueError('GPU03 allocation required for complete coupled future-noise check')
    work.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', training_performed=False, scientific_claim_qualified=False,
                  native_initial_noise_blocks=1, complete_native_executions=0,
                  job_id=os.environ['SLURM_JOB_ID'], started_utc=datetime.now(timezone.utc).isoformat())
    graph=None
    code=2
    try:
        proof=json.loads((canary/'delivery.json').read_text())
        checks=json.loads((canary/'results/report.json').read_text())
        build_proof=json.loads((native_build/'build.json').read_text())
        if (proof['status']!='PASS' or proof['exit_code']!=0 or checks['status']!='PASS'
                or checks.get('future_poisson_rng_interface_verified') is not True
                or proof['native_library_sha256']!=build_proof['library_sha256']):
            raise ValueError('passed bound future-noise native canary required')
        plan,manifest=stage_coupled_inputs(repo,work,native_build,producer_software)
        record['plan']=plan
        env=dict(os.environ,PYTHONPATH=str(work)+':'+str(work/'Libs'),NESTGPU_LIB=str(work/'libnestgpukernel.so'),
                 PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TMPDIR=str(work))
        resource_command=[sys.executable,'-m','Libs.population_dynamics.resource_probe','--output',str(work/'resource.json'),
                          '--required-memory-gib','32']
        with (work/'resource.log').open('w') as log:
            health=subprocess.run(resource_command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
        if health.returncode:
            raise RuntimeError('coupled future-noise resource gate failed')
        graph=Path('/dev/shm')/('neuronsim-future-noise-graph-'+os.environ['SLURM_JOB_ID'])
        if shutil.disk_usage('/dev/shm').free<plan['graph_bytes']+4*1024**3:
            raise ValueError('insufficient RAM scratch for fixed coupled graph')
        stage_native_graph(repo,work,graph,manifest)
        settings=dict(plan['settings'])
        boundary_tick=10000
        reset_tick=10015
        settings['state_observation_ticks']=sorted(set(settings['state_observation_ticks'])|{boundary_tick,reset_tick})
        source_case=next(c for c in json.loads((work/'inputs/design.json').read_text())['cases'] if c['case_id']==COUPLED_CASE)
        settings['pulse_control']=source_case['settings']['pulse_control']
        settings['live_pulse_control']=False
        runs={}
        for name,seeds in (('unchanged',None),('future_a',list(range(87001,87009))),
                           ('future_a_replay',list(range(87001,87009))),('future_b',list(range(87011,87019)))):
            current=dict(settings)
            if seeds is not None:
                current['future_background_noise']=dict(boundary_time_ms=1000.,seeds_per_population=seeds)
            root=work/'executions'/name;root.mkdir(parents=True)
            (root/'execution.yaml').write_text(yaml.safe_dump(current,sort_keys=False))
            command=[sys.executable,'-m','Libs.population_dynamics.nest_gpu_spatial','--graph',str(graph),
                     '--config',str(root/'execution.yaml'),'--output',str(root/'native')]
            with (root/'execution.log').open('w') as log:
                result=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
            native=json.loads((root/'native/report.json').read_text())
            if result.returncode or native['status']!='EXECUTED':
                raise RuntimeError('complete coupled future-noise execution failed: '+name)
            if seeds is not None and (len(native['future_background_noise'])!=8
                    or sum(r['outgoing_edges_verified'] for r in native['future_background_noise'])!=100000
                    or any(r['reset_tick']!=reset_tick for r in native['future_background_noise'])):
                raise ValueError('all source background generators must switch at the declared native tick')
            runs[name]=root/'native'
            record['complete_native_executions']+=1
        comparisons=[]
        for filename in plan['reference_arrays']:
            with np.load(work/'reference'/filename) as left,np.load(runs['unchanged']/filename) as right:
                for key in left.files:
                    comparisons.append(dict(artifact=filename,array=key,exact=np.array_equal(left[key],right[key])))
        if not all(row['exact'] for row in comparisons):
            raise ValueError('new library unchanged full episode differs from original accepted source')
        for filename in [p.name for p in runs['future_a'].iterdir() if p.suffix in ('.npz','.npy')]:
            if filename.endswith('.npy'):
                exact=np.array_equal(np.load(runs['future_a']/filename),np.load(runs['future_a_replay']/filename))
            else:
                with np.load(runs['future_a']/filename) as left,np.load(runs['future_a_replay']/filename) as right:
                    exact=set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k]) for k in left.files)
            if not exact:raise ValueError('same future seed complete source replay differs: '+filename)
        baseline=runs['unchanged']
        for name in ('future_a','future_b'):
            for tick in (boundary_tick,reset_tick):
                filename=f'state_observation_{tick:09d}.npz'
                with np.load(baseline/filename) as left,np.load(runs[name]/filename) as right:
                    if not all(np.array_equal(left[k],right[k]) for k in left.files):
                        raise ValueError('full unit state changed before future arrivals: '+name)
            with np.load(baseline/'spikes.npz') as left,np.load(runs[name]/'spikes.npz') as right:
                for key in ('spike_ticks','spike_unit_ids'):
                    if not np.array_equal(left[key][left['spike_ticks']<=reset_tick],right[key][right['spike_ticks']<=reset_tick]):
                        raise ValueError('committed full source event prefix changed: '+name)
        with np.load(runs['future_a']/'spikes.npz') as left,np.load(runs['future_b']/'spikes.npz') as right:
            future_changed=any(not np.array_equal(left[k][left['spike_ticks']>reset_tick],right[k][right['spike_ticks']>reset_tick])
                               for k in ('spike_ticks','spike_unit_ids'))
        if not future_changed:raise ValueError('distinct future noise streams produced identical complete source events')
        record.update(status='PASS',native_library_sha256=build_proof['library_sha256'],
            original_complete_episode_regression_exact=True,same_future_seed_full_array_replay_exact=True,
            full100000_unit_boundary_states_exact=True,committed_event_prefix_exact=True,
            distinct_future_seeds_change_source_events=True,boundary_tick=boundary_tick,reset_tick=reset_tick,
            first_resampled_arrival_tick=reset_tick+1,original_comparisons=comparisons,
            conditional_scope='one replayed microscopic initial/background realization; hidden-state posterior and distributional fidelity unqualified')
        code=0
    except Exception as error:
        record.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
    finally:
        record.update(exit_code=code,ended_utc=datetime.now(timezone.utc).isoformat())
        archive(work,target,record)
        if graph is not None and code==0:shutil.rmtree(graph)
    return code


def require_future_qualification(coupled, library_sha256):
    proof = json.loads((coupled/'delivery.json').read_text())
    required = dict(status='PASS', exit_code=0, native_library_sha256=library_sha256,
                    complete_native_executions=4, boundary_tick=10000, reset_tick=10015,
                    first_resampled_arrival_tick=10016,
                    original_complete_episode_regression_exact=True,
                    same_future_seed_full_array_replay_exact=True,
                    full100000_unit_boundary_states_exact=True,
                    committed_event_prefix_exact=True, distinct_future_seeds_change_source_events=True)
    if (any(proof.get(key) != value for key, value in required.items())
            or proof.get('plan', {}).get('case_id') != COUPLED_CASE
            or proof['plan'].get('core_neurons') != 100000
            or proof['plan'].get('duration_ms') != 6000):
        raise ValueError('exact full coupled future-noise qualification required')
    return proof


def future_ensemble_seeds(config):
    required = dict(scope='fixed_microstate_conditional_development', case_id=COUPLED_CASE,
                    boundary_time_ms=1000., episode_duration_ms=6000., joint_samples=16,
                    populations=8, formal_evaluation=False,
                    matched_forecast_configuration=MATCHED_FORECAST_CONFIG)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('declared sixteen-future complete native development protocol required')
    base, stride = config['future_background_seed_base'], config['future_background_seed_stride']
    if type(base) is not int or type(stride) is not int or base < 1 or stride < 8:
        raise ValueError('positive integer future seed base and disjoint eight-source stride required')
    seeds = [[base + index*stride + population for population in range(8)] for index in range(16)]
    if max(value for row in seeds for value in row) >= 2**31:
        raise ValueError('future Poisson seeds must fit signed int31 support')
    return seeds


def stage_future_ensemble_inputs(repo, work, coupled, producer_software, plan):
    """Reuse the qualified boundary, original observation map and fitted scale."""
    proof = require_future_qualification(coupled, plan['native_library_sha256'])
    if proof['plan']['original_report_sha256'] != plan['original_report_sha256']:
        raise ValueError('future qualification and ensemble reference differ')
    qualified_rows = manifest_rows(proof)
    for tick in (10000, 10015):
        name = f'state_observation_{tick:09d}.npz'
        relative = 'executions/unchanged/native/'+name
        copy_bound(coupled/relative, work/'qualified_boundary'/name, qualified_rows[relative])
    source = repo/COHORT
    source_rows = manifest_rows(json.loads((source/'delivery.json').read_text()))
    for name in ('summary.json', 'spatial_fields.npz'):
        relative = 'train_fields/'+COUPLED_CASE+'/'+name
        copy_bound(source/relative, work/'reference_fields'/name, source_rows[relative])
    seed41 = repo/ACCEPTED_SEED41
    seed_proof = json.loads((seed41/'delivery.json').read_text())
    if seed_proof['status'] != 'PASS' or seed_proof['exit_code'] != 0:
        raise ValueError('complete accepted seed41 fitted-scale delivery required')
    scale_name = 'forecasts/fitting_field_scale.npz'
    copy_bound(seed41/scale_name, work/'inputs/fitting_field_scale.npz', manifest_rows(seed_proof)[scale_name])
    software_rows = manifest_rows(json.loads((producer_software/'delivery.json').read_text()))
    for name in (FUTURE_ENSEMBLE_CONFIG, MATCHED_FORECAST_CONFIG):
        copy_bound(producer_software/name, work/name, software_rows[name])
    config = yaml.safe_load((work/FUTURE_ENSEMBLE_CONFIG).read_text())
    seeds = future_ensemble_seeds(config)
    return seeds, dict(configuration=config, qualification_delivery_sha256=digest(coupled/'delivery.json'),
                       reference_field_summary_sha256=digest(work/'reference_fields/summary.json'),
                       fitting_scale_sha256=digest(work/'inputs/fitting_field_scale.npz'),
                       fitting_scale_source=ACCEPTED_SEED41+'/'+scale_name,
                       matched_forecast_config_sha256=digest(work/MATCHED_FORECAST_CONFIG))


def execute_future_ensemble(repo, work, target, native_build, coupled, producer_software):
    """Execute a measured fixed-microstate cost point without model resampling."""
    if (os.environ.get('SLURMD_NODENAME') != 'gpu03' or not os.environ.get('SLURM_JOB_GPUS')
            or not os.environ.get('SLURM_JOB_ID')):
        raise ValueError('GPU03 allocation required for native future ensemble')
    work.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(status='RUNNING', job_id=os.environ['SLURM_JOB_ID'],
                  started_utc=datetime.now(timezone.utc).isoformat(), native_executions=[],
                  hardware=dict(host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                                cpus_requested=os.environ.get('SLURM_CPUS_PER_TASK'),
                                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  training_performed=False, learned_generation_repeated=False,
                  end_to_end_cost_qualified=False, matched_fidelity_qualified=False,
                  same_hardware_comparison_qualified=False, speedup_qualified=False,
                  scientific_claim_qualified=False)
    graph = None
    code = 2
    try:
        plan, manifest = stage_coupled_inputs(repo, work, native_build, producer_software)
        seeds, binding = stage_future_ensemble_inputs(repo, work, coupled, producer_software, plan)
        record.update(plan=plan, ensemble_binding=binding,
                      input_and_software_staging_seconds=time.perf_counter()-started)
        env = dict(os.environ, PYTHONPATH=str(work)+':'+str(work/'Libs'),
                   NESTGPU_LIB=str(work/'libnestgpukernel.so'), PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TMPDIR=str(work))
        resource_command = [sys.executable, '-m', 'Libs.population_dynamics.resource_probe',
                            '--output', str(work/'resource.json'), '--required-memory-gib', '32']
        with (work/'resource.log').open('w') as log:
            health = subprocess.run(resource_command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        if health.returncode:
            raise RuntimeError('native future ensemble resource gate failed')
        graph = Path('/dev/shm')/('neuronsim-future-ensemble-graph-'+os.environ['SLURM_JOB_ID'])
        if shutil.disk_usage('/dev/shm').free < plan['graph_bytes']+4*1024**3:
            raise ValueError('insufficient RAM scratch for fixed coupled graph')
        stage_start = time.perf_counter()
        stage_native_graph(repo, work, graph, manifest)
        record['graph_staging_seconds'] = time.perf_counter()-stage_start
        settings = dict(plan['settings'])
        settings['state_observation_ticks'] = sorted(set(settings['state_observation_ticks'])|{10000, 10015})
        source_case = next(c for c in json.loads((work/'inputs/design.json').read_text())['cases']
                           if c['case_id'] == COUPLED_CASE)
        settings['pulse_control'] = source_case['settings']['pulse_control']
        settings['live_pulse_control'] = False
        branches = []
        for index, vector in enumerate(seeds):
            root = work/'executions'/f'draw_{index:02d}'
            root.mkdir(parents=True)
            current = dict(settings, future_background_noise=dict(boundary_time_ms=1000., seeds_per_population=vector))
            (root/'execution.yaml').write_text(yaml.safe_dump(current, sort_keys=False))
            command = [sys.executable, '-m', 'Libs.population_dynamics.nest_gpu_spatial',
                       '--graph', str(graph), '--config', str(root/'execution.yaml'), '--output', str(root/'native')]
            execution_start = time.perf_counter()
            with (root/'execution.log').open('w') as log:
                result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            cost = dict(draw=index, seeds_per_population=vector, command=command,
                        exit_code=result.returncode, process_seconds=time.perf_counter()-execution_start)
            record['native_executions'].append(cost)
            native = json.loads((root/'native/report.json').read_text())
            if result.returncode or native['status'] != 'EXECUTED':
                raise RuntimeError('complete native ensemble execution failed: '+str(index))
            cost.update(native_function_seconds=native['elapsed_seconds'],
                        graph_import_seconds=native['import_seconds'],
                        simulation_seconds=native['simulation_seconds'], calibration_seconds=native['calibration_seconds'],
                        native_peak_process_rss_bytes=int(native['peak_process_rss_mib']*1024**2))
            branches.append(str(root/'native'))
            print(json.dumps(dict(status='EXECUTING_NATIVE_FUTURES', complete_draws=len(branches), required_draws=16)), flush=True)
            with (root/'resource.log').open('w') as log:
                health = subprocess.run(resource_command[:3]+['--output', str(root/'resource.json'),
                    '--required-memory-gib', '32'], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            if health.returncode:
                raise RuntimeError('native future ensemble resource gate failed after draw '+str(index))
        (work/'branches.json').write_text(json.dumps(branches, indent=2)+'\n')
        command = [sys.executable, '-m', 'Libs.population_dynamics.native_future_ensemble',
                   '--graph', str(graph), '--reference-episode', str(work/'reference'),
                   '--reference-fields', str(work/'reference_fields'), '--branches', str(work/'branches.json'),
                   '--fitting-scale', str(work/'inputs/fitting_field_scale.npz'),
                   '--config', str(work/MATCHED_FORECAST_CONFIG), '--output', str(work/'analysis'),
                   '--committed-state-reference', str(work/'qualified_boundary')]
        analysis_start = time.perf_counter()
        with (work/'analysis.log').open('w') as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(analysis_command=command, analysis_exit_code=result.returncode,
                      analysis_process_seconds=time.perf_counter()-analysis_start)
        if result.returncode:
            raise RuntimeError('native future ensemble readout failed')
        analysis = json.loads((work/'analysis/analysis.json').read_text())
        if (analysis['status'] != 'DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT'
                or analysis['draws'] != 16 or analysis['generated_unit_count_values'] != 160000000
                or analysis['full_unit_counts_saved'] is not True
                or analysis['all_committed_events_and_states_exact'] is not True
                or analysis['all_tc_programs_exact'] is not True
                or analysis['branch_source_identity']['native_library_sha256'] != plan['native_library_sha256']):
            raise ValueError('complete fixed-microstate all-unit ensemble readout required')
        record.update(status='PASS', complete_native_executions=16,
                      analysis_status=analysis['status'], full_unit_counts_saved=True,
                      native_process_seconds=sum(row['process_seconds'] for row in record['native_executions']),
                      native_function_seconds=sum(row['native_function_seconds'] for row in record['native_executions']),
                      simulated_biological_ms=16*6000, target_output_biological_ms=16*500,
                      generated_unit_count_values=16*100*100000,
                      cost_scope='measured single fixed-microstate condition; rebuild/replay and full6s output per draw; process imports included; training, queue wait and comparative fidelity unqualified')
        code = 0
    except Exception as error:
        record.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
    finally:
        record.update(exit_code=code, pipeline_before_archive_seconds=time.perf_counter()-started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        archive_start = time.perf_counter()
        archive(work, target, record)
        record.update(archive_seconds=time.perf_counter()-archive_start,
                      task_through_archive_seconds=time.perf_counter()-started)
        (target/'delivery.json').write_text(json.dumps(record, indent=2)+'\n')
        if graph is not None and code == 0:
            shutil.rmtree(graph)
    return code


def require_native_qualification(canary, coupled, library_sha256):
    proof = json.loads((canary/'delivery.json').read_text())
    checks = json.loads((canary/'results/report.json').read_text())
    if (proof['status'] != 'PASS' or proof['exit_code'] != 0 or checks['status'] != 'PASS'
            or not checks['all_six_cases_executed'] or not checks['no_mutation_original_library_regression_verified']
            or proof['native_library_sha256'] != library_sha256):
        raise ValueError('complete native canary and original-library regression required')
    full = json.loads((coupled/'delivery.json').read_text())
    comparisons = json.loads((coupled/'comparison.json').read_text())
    if (full['status'] != 'PASS' or full['exit_code'] != 0 or comparisons['status'] != 'PASS'
            or full['plan']['native_library_sha256'] != library_sha256
            or full['plan']['core_neurons'] != 100000 or full['plan']['duration_ms'] != 6000
            or full['plan']['case_id'] != COUPLED_CASE
            or not comparisons['comparisons'] or not all(r['exact'] for r in comparisons['comparisons'])):
        raise ValueError('exact complete coupled native replay required before candidate execution')


def stage_pulse_search(search, work, plan):
    """Consume the completed search and replay without repeating inference."""
    proof = json.loads((search/'delivery.json').read_text())
    if (proof['status'] != 'PASS' or proof['exit_code'] != 0 or proof['training_seeds'] != [41,42,43]
            or not proof['saved_array_replay_completed'] or proof['training_performed']
            or proof['mechanistic_simulation_performed']):
        raise ValueError('completed three-seed inference and independent saved-array replay required')
    rows = manifest_rows(proof)
    names = ['results/analysis.json', 'replay.json']
    names += [f'results/seed_{seed}/{name}' for seed in [41,42,43] for name in
              ['analysis.json','reference.npz','omit.npz','first_future_pulse.npz','original_future_pulses.npz']]
    for name in names:
        copy_bound(search/name, work/'inputs/search'/name, rows[name])
    aggregate = json.loads((work/'inputs/search/results/analysis.json').read_text())
    replay = json.loads((work/'inputs/search/replay.json').read_text())
    if (aggregate['status'] != 'THREE_SEED_DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED'
            or aggregate['completed_seeds'] != [41,42,43] or aggregate['candidate_results'] != 9
            or replay['status'] != 'INDEPENDENT_THREE_SEED_SAVED_ARRAY_PULSE_SEARCH_REPLAYED'
            or replay['completed_seeds'] != [41,42,43]):
        raise ValueError('complete three-seed candidate and replay denominator required')
    reports = {seed:json.loads((work/f'inputs/search/results/seed_{seed}/analysis.json').read_text())
               for seed in [41,42,43]}
    reference = json.loads((work/'reference/report.json').read_text())
    for seed, report in reports.items():
        binding = report['native_binding']
        identity = binding['native_source_identity']
        if (binding['episode_report_sha256'] != plan['original_report_sha256']
                or binding['initial_parameters_sha256'] != digest(work/'reference/initial_parameters.npz')
                or binding['spikes_sha256'] != digest(work/'reference/spikes.npz')
                or identity['native_library_sha256'] != reference['native_library_sha256']
                or identity['graph_manifest_sha256'] != plan['graph_manifest_sha256']
                or binding['field_summary_sha256'] != digest(work/'reference_fields/summary.json')
                or binding['spatial_fields_sha256'] != digest(work/'reference_fields/spatial_fields.npz')):
            raise ValueError(f'seed {seed} search source differs from native replay source')
        row = next(row for row in aggregate['seeds'] if row['training_seed'] == seed)
        if row['analysis_sha256'] != digest(work/f'inputs/search/results/seed_{seed}/analysis.json'):
            raise ValueError('aggregate search report identity changed')
    source = search/'delivery.json'
    copy_bound(source, work/'prerequisites/search_delivery.json',dict(bytes=source.stat().st_size,sha256=digest(source)))
    return reports


def stage_pulse_software(software, work):
    proof = json.loads((software/'delivery.json').read_text())
    if proof['status'] != 'PASS' or proof['exit_code'] != 0:
        raise ValueError('accepted native pulse execution software required')
    for name, row in manifest_rows(proof).items():
        if name.startswith('Libs/') and name.endswith('.py'):
            target = work/name
            if target.exists():
                if digest(target) != row['sha256']:
                    raise ValueError('native producer differs from tested pulse software: '+name)
            else:
                copy_bound(software/name, target, row)
    source = software/'delivery.json'
    copy_bound(source, work/'prerequisites/pulse_software.json',dict(bytes=source.stat().st_size,sha256=digest(source)))
    return proof


def execute_pulse_search(repo, work, target, native_build, canary, coupled, search, software, producer_software=None):
    """Execute all three saved native schedules in serial fresh processes."""
    if (os.environ.get('SLURMD_NODENAME') != 'gpu03' or not os.environ.get('SLURM_JOB_ID')
            or not os.environ.get('SLURM_JOB_GPUS') or Path('/tmp/neuronsim') not in work.parents):
        raise ValueError('allocated GPU03 and node-local scratch required')
    if target.exists():
        raise FileExistsError('existing native pulse evidence must be preserved')
    work.mkdir(parents=True, exist_ok=False)
    record = dict(status='PREPARING', job_id=os.environ['SLURM_JOB_ID'],
                  started_utc=datetime.now(timezone.utc).isoformat(), training_performed=False,
                  scientific_claim_qualified=False, independent_intervention_qualified=False,
                  success_tolerance_qualified=False, native_future_noise_pairing_verified=False,
                  mechanistic_simulation_performed=False, training_seeds=[41,42,43],
                  native_initial_noise_blocks=1, native_unique_candidates_expected=3, native_candidates_completed=0,
                  selection_used_for_checkpoint_selection=True, candidates=[], comparisons=[])
    graph = None
    code = 2
    try:
        plan, manifest = (stage_coupled_inputs(repo, work, native_build) if producer_software is None
                          else stage_coupled_inputs(repo, work, native_build, producer_software))
        require_native_qualification(canary, coupled, plan['native_library_sha256'])
        proof = stage_pulse_software(software, work)
        record['software_job_id'] = proof['job_id']
        sys.path.insert(0, str(work/'Libs'))
        from population_dynamics.mesocircuit_continuation import prepare_pulse_protocol
        from population_dynamics.native_pulse_programs import native_candidate_protocols
        from population_dynamics.native_pulse_comparison import compare_candidate
        from population_dynamics.native_renewal_data import NativeRenewalEpisode
        # Restore the exact archived field view used by all search checkpoints.
        source = repo/COHORT
        index = manifest_rows(json.loads((source/'delivery.json').read_text()))
        for name in ['summary.json','spatial_fields.npz']:
            relative = 'train_fields/'+COUPLED_CASE+'/'+name
            copy_bound(source/relative, work/'reference_fields'/name, index[relative])
        reports = stage_pulse_search(search, work, plan)
        case = next(r for r in json.loads((work/'inputs/design.json').read_text())['cases'] if r['case_id']==COUPLED_CASE)
        programs, selected = native_candidate_protocols(reports, case, dict(manifest, graph_manifest_sha256=plan['graph_manifest_sha256']))
        original = NativeRenewalEpisode(work/'reference',work/'reference_fields')
        plan = dict(plan,decision_time_ms=1000.,pending_committed_tc_arrival_ms=None,
                    settings=dict(case['settings'],source_rng_library=str(work/'source_rng.so')),
                    reason='execute every saved three-seed development candidate on one coupled source initial/noise block')
        record.update(plan=plan, selected_by_training_seed=selected, programs=programs,
                      source_search_delivery_sha256=digest(work/'prerequisites/search_delivery.json'))
        env = dict(os.environ, PYTHONPATH=str(work/'Libs'), NESTGPU_LIB=str(work/'libnestgpukernel.so'),
                   PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (work/'resource.log').open('w') as stream:
            health = subprocess.run(['python3','-m','population_dynamics.resource_probe','--output',str(work/'resource.json'),
                                     '--required-memory-gib','32'],cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if health.returncode:
            raise RuntimeError('native pulse resource gate failed')
        if shutil.disk_usage('/dev/shm').free < plan['graph_bytes'] + 4*1024**3:
            raise ValueError('insufficient RAM scratch for coupled graph')
        graph = Path('/dev/shm')/('neuronsim-live-control-graph-'+os.environ['SLURM_JOB_ID'])
        stage_native_graph(repo, work, graph, manifest)
        fields_config = work/'field_projection.yaml'
        fields_config.write_text(yaml.safe_dump(dict(bin_ms=5.,cells_per_axis=4,start_ms=0.,pulse_window_ms=10.)))
        record['status'] = 'RUNNING'
        for name, program in programs.items():
            root = work/'candidates'/name
            root.mkdir(parents=True)
            settings = dict(case['settings'],source_rng_library=str(work/'source_rng.so'),live_pulse_control=True,pulse_control=program)
            config = root/'execution.yaml'
            config.write_text(yaml.safe_dump(settings,sort_keys=False))
            commands = [
                ['python3','-m','population_dynamics.nest_gpu_spatial','--graph',str(graph),'--config',str(config),'--output',str(root/'native')],
                ['python3','-m','population_dynamics.spatial_activity','--graph',str(graph),'--episode',str(root/'native'),
                 '--config',str(fields_config),'--output',str(root/'fields')],
            ]
            outcome = dict(candidate=name,status='RUNNING',commands=[],selected_by_training_seeds=[s for s in selected if selected[s]==name])
            record['candidates'].append(outcome)
            for index, command in enumerate(commands):
                if index == 0:
                    record['mechanistic_simulation_performed'] = True
                with (root/f'phase_{index}.log').open('w') as stream:
                    result = subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
                outcome['commands'].append(dict(command=command,exit_code=result.returncode))
                if result.returncode:
                    raise RuntimeError('native pulse '+name+' phase '+str(index)+' failed')
            native_report = json.loads((root/'native/report.json').read_text())
            if (native_report['status'] != 'EXECUTED' or not native_report['native_live_control']['future_schedule_replaced']):
                raise ValueError('complete unchanged initial state and actual live mutation required')
            with np.load(root/'native/initial_parameters.npz',allow_pickle=False) as actual_initial, np.load(work/'reference/initial_parameters.npz',allow_pickle=False) as source_initial:
                if (set(actual_initial.files) != set(source_initial.files)
                        or any(actual_initial[k].dtype != source_initial[k].dtype or not np.array_equal(actual_initial[k],source_initial[k]) for k in source_initial.files)):
                    raise ValueError('candidate initial native parameters differ from the archived source')
            actual = NativeRenewalEpisode(root/'native',root/'fields',expected_unit_ids=original.unit_ids,
                                          expected_unit_field_index=original.unit_field_index)
            protocol = prepare_pulse_protocol(manifest['resolved_source_network'],
                dict(sim_resolution=.1,t_presim=0.,t_sim=6000.),program)
            if not np.array_equal(actual.pulse_ticks,np.asarray(protocol.effective_ticks,dtype=np.int64)+protocol.delay_ticks):
                raise ValueError('actual native TC arrivals differ from the saved candidate program')
            outcome['actual_tc_arrival_ticks_verified'] = actual.pulse_ticks.tolist()
            for seed in [41,42,43]:
                saved = work/f'inputs/search/results/seed_{seed}'
                with np.load(saved/(name+'.npz'),allow_pickle=False) as prediction, np.load(saved/'reference.npz',allow_pickle=False) as reference:
                    comparison, arrays = compare_candidate(actual,original,dict(prediction),dict(reference))
                comparison.update(candidate=name,training_seed=seed,selected=selected[seed]==name,
                                  source_prediction_sha256=digest(saved/(name+'.npz')),native_report_sha256=digest(root/'native/report.json'))
                if seed == 41:
                    np.savez_compressed(root/'actual_response.npz',**arrays)
                (root/f'comparison_seed_{seed}.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
                record['comparisons'].append(comparison)
            outcome.update(status='EXECUTED_AND_COMPARED',native_report_sha256=digest(root/'native/report.json'))
            record['native_candidates_completed'] += 1
        if len(record['candidates']) != 3 or len(record['comparisons']) != 9:
            raise ValueError('all native candidate and training seed comparisons required')
        code = 0
        record['status'] = 'EXECUTED_DEVELOPMENT'
    except Exception as error:
        record.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
    finally:
        # Reference inputs and libraries already have authoritative archives.
        # Copy only declared new evidence; retain partial native outputs on failure.
        (work/'inputs').mkdir(exist_ok=True)
        for name in ['reference','reference_fields']:
            source = work/name
            if source.exists():
                source.rename(work/'inputs'/name)
        for name in ['libnestgpukernel.so','nestgpu.py','source_rng.so']:
            source = work/name
            if source.exists():
                source.rename(work/'inputs'/name)
        record['exit_code'] = code
        archive(work,target,record)
        if graph is not None and code == 0:
            shutil.rmtree(graph)
    return code


def replay_coupled(repo, work, target, native_build, prepare_only, canary, producer_software=None):
    node = os.environ.get('SLURMD_NODENAME')
    if not os.environ.get('SLURM_JOB_ID') or (prepare_only and node != 'cpu04') or (not prepare_only and node != 'gpu03'):
        raise ValueError('CPU04 preparation or allocated GPU03 native replay required')
    if not prepare_only and not os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('native replay requires a GPU allocation')
    target.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    code = 2
    record = dict(status='PREPARING', job_id=os.environ['SLURM_JOB_ID'], work=str(work),
                  scientific_claim_qualified=False, mechanistic_simulation_performed=False)
    graph = None
    try:
        record['plan'], manifest = stage_coupled_inputs(repo, work, native_build, producer_software)
        if prepare_only:
            # Qualification preparation verifies the real identities and causal
            # protocol. It does not import graph payloads or execute a simulator.
            sys.path.insert(0, str(work))
            from Libs.population_dynamics.mesocircuit_continuation import prepare_pulse_protocol
            settings = record['plan']['settings']
            protocol = prepare_pulse_protocol(manifest['resolved_source_network'],
                dict(sim_resolution=.1, t_presim=50., t_sim=5950.), settings['pulse_control'])
            if (len(protocol.committed_ticks) != 3 or protocol.committed_ticks[-1] != protocol.decision_tick
                    or protocol.effective_ticks != protocol.initial_ticks):
                raise ValueError('full replay control must retain boundary emission and identical effective schedule')
            record.update(status='PREPARED', protocol=dict(protocol.record(), boundary_phase='planned: after continuous decision-tick step, before device schedule mutation'))
            code = 0
        else:
            proof = json.loads((canary/'delivery.json').read_text())
            qualified = json.loads((canary/'results/report.json').read_text())
            if (proof['status'] != 'PASS' or proof['exit_code'] != 0 or qualified['status'] != 'PASS'
                    or not qualified['all_six_cases_executed'] or not qualified['no_mutation_original_library_regression_verified']
                    or proof['native_library_sha256'] != record['plan']['native_library_sha256']):
                raise ValueError('all native interface and original-library comparisons must pass before coupled replay')
            env = dict(os.environ, PYTHONPATH=str(work), NESTGPU_LIB=str(work/'libnestgpukernel.so'),
                       PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            with (work/'resource.log').open('w') as stream:
                health = subprocess.run(['python3', '-m', 'Libs.population_dynamics.resource_probe', '--output', str(work/'resource.json'),
                                         '--required-memory-gib', '32'], cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if health.returncode:
                raise RuntimeError('native replay resource gate failed')
            # Use RAM scratch for the 15.7GB fixed graph, preserving root disk space.
            graph = Path('/dev/shm')/('neuronsim-live-control-graph-'+os.environ['SLURM_JOB_ID'])
            if shutil.disk_usage('/dev/shm').free < record['plan']['graph_bytes'] + 4*1024**3:
                raise ValueError('insufficient RAM scratch for fixed coupled graph')
            stage_native_graph(repo, work, graph, manifest)
            record.update(status='RUNNING', graph_payload_staged=True, mechanistic_simulation_performed=True)
            (target/'delivery.json').write_text(json.dumps(record, indent=2)+'\n')
            command = ['python3', '-m', 'Libs.population_dynamics.nest_gpu_spatial', '--graph', str(graph),
                       '--config', str(work/'execution.yaml'), '--output', str(work/'results')]
            record['command'] = command
            with (work/'execution.log').open('w') as stream:
                result = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('native coupled replay execution failed')
            native = json.loads((work/'results/report.json').read_text())
            if native['status'] != 'EXECUTED' or not native['native_live_control']['future_schedule_replaced']:
                raise ValueError('native replay must execute an actual future schedule update')
            comparison = compare_coupled_replay(work, record['plan']['reference_arrays'])
            code = 0 if comparison['status'] == 'PASS' else 2
            record.update(status='PASS' if code == 0 else 'FAILED', comparison_status=comparison['status'])
    except Exception as error:
        code = code or 2
        record.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
    finally:
        for name in ['plan.json', 'execution.yaml', 'comparison.json', 'resource.json', 'resource.log', 'execution.log']:
            if (work/name).exists():
                shutil.copyfile(work/name, target/name)
        for name in ['results', 'Libs']:
            if (work/name).exists():
                shutil.copytree(work/name, target/name)
        record.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat())
        (target/'delivery.json').write_text(json.dumps(record, indent=2)+'\n')
        if graph is not None and code == 0:
            shutil.rmtree(graph)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-build', type=Path)
    parser.add_argument('--future-poisson-rng-build', action='store_true',
                        help='Build the uniform-delay future Poisson RNG interface')
    parser.add_argument('--future-poisson-rng-canary', action='store_true',
                        help='Check preserved committed input and independent future RNG seeds')
    parser.add_argument('--future-poisson-rng-coupled', action='store_true',
                        help='Check one full coupled source boundary and independent future seeds')
    parser.add_argument('--future-poisson-rng-ensemble', action='store_true',
                        help='Measure sixteen distinct full-unit futures of one fixed microscopic state')
    parser.add_argument('--coupled-replay', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--canary', type=Path)
    parser.add_argument('--canary-software', type=Path, help='Accepted native interface-check software')
    parser.add_argument('--producer-software', type=Path, help='Accepted manifest-bound coupled native producer; defaults to the historical producer')
    parser.add_argument('--pulse-search', type=Path)
    parser.add_argument('--coupled-qualification', type=Path)
    parser.add_argument('--pulse-software', type=Path)
    args = parser.parse_args()
    if args.future_poisson_rng_ensemble:
        if (not args.native_build or not args.coupled_qualification or not args.producer_software
                or args.coupled_replay or args.pulse_search or args.prepare_only or args.canary
                or args.future_poisson_rng_canary or args.future_poisson_rng_build or args.future_poisson_rng_coupled):
            parser.error('native future ensemble requires build, full future qualification and producer software only')
        raise SystemExit(execute_future_ensemble(args.repo, args.work, args.output, args.native_build,
                                                args.coupled_qualification, args.producer_software))
    if args.future_poisson_rng_coupled:
        if (not args.native_build or not args.canary or not args.producer_software
                or args.coupled_replay or args.pulse_search or args.prepare_only
                or args.future_poisson_rng_canary or args.future_poisson_rng_build):
            parser.error('coupled future noise requires build, canary and producer software only')
        raise SystemExit(check_coupled_future_noise(args.repo,args.work,args.output,args.native_build,
                                                  args.canary,args.producer_software))
    if args.future_poisson_rng_build and (args.native_build or args.pulse_search):
        parser.error('--future-poisson-rng-build is exclusive with native execution modes')
    if args.future_poisson_rng_canary and (not args.native_build or args.coupled_replay
                                         or args.pulse_search or args.prepare_only):
        parser.error('--future-poisson-rng-canary requires a native build and isolated canary mode')
    if args.pulse_search:
        if not args.native_build or not args.canary or not args.coupled_qualification or not args.pulse_software or args.prepare_only or args.coupled_replay:
            parser.error('--pulse-search requires native build, canary, coupled qualification and pulse software; execution only')
        raise SystemExit(execute_pulse_search(args.repo,args.work,args.output,args.native_build,args.canary,
                                             args.coupled_qualification,args.pulse_search,args.pulse_software,args.producer_software))
    if args.native_build:
        raise SystemExit(check(args.repo, args.work, args.output, args.native_build,
                              args.coupled_replay, args.prepare_only, args.canary, args.canary_software,
                              args.producer_software, args.future_poisson_rng_canary))
    print(json.dumps(build(args.repo, args.work, args.output, args.future_poisson_rng_build)))


if __name__ == '__main__':
    main()

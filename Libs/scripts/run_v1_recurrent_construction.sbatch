#!/bin/bash
#SBATCH --job-name=NrnSim_V1RecurrentConstruction
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_recurrent-%j.log
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
srun --ntasks=1 --cpu-bind=none "${NRNSIM_V1_PYTHON:?Set validated scientific Python}" -B - "$@" "$0" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import traceback

import yaml

repo, config_path, qualification = map(Path, sys.argv[1:4])
mode = sys.argv[4] if len(sys.argv) == 6 else 'full'
launcher = Path(sys.argv[-1])
job = os.environ['SLURM_JOB_ID']
node = os.environ['SLURMD_NODENAME']
if mode not in ('full', 'canary') or node not in ('cpu03', 'cpu04') or os.environ.get('SLURM_JOB_GPUS'):
    raise ValueError('V1_RECURRENT_CPU_WITHOUT_GPU_REQUIRED')
if mode == 'full' and node != 'cpu03':
    raise ValueError('V1_FULL_RECURRENT_CONSTRUCTION_CPU03_REQUIRED')
destination = repo / 'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/recurrent_construction' / ('run_' + job)
if destination.exists():
    raise FileExistsError(destination)
modules = ('v1_circuit_build', 'v1_circuit_source', 'v1_graph_identity', 'v1_source_conversion',
           'provenance', 'resource_probe', 'v1_recurrent_runtime')
source_names = ('original_build_network', 'original_node_functions', 'original_edge_functions',
    'original_node_parameters', 'original_edge_parameters', 'original_connection_parameters',
    'original_lgn_connection_parameters', 'original_glif_node_parameters', 'original_bio_to_glif_map',
    'weight_semantics/legacy_glif_converter', 'weight_semantics/legacy_recurrent_table')
receipt = dict(status='PREPARING', job_id=job, node=node, mode=mode,
               scientific_qualification=False, production_episodes=0,
               full_recurrent_realizations_generated=0, commands=[])
code = 1

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

with tempfile.TemporaryDirectory(prefix='neuronsim-v1-recurrent-' + job + '.', dir='/tmp') as temporary:
    work = Path(temporary)
    try:
        proof = json.loads((qualification / 'validation.json').read_text())
        comparison = json.loads((qualification / 'comparison/report.json').read_text())
        canary = json.loads((qualification / 'canary/report.json').read_text())
        if (proof['status'] != 'PASS' or proof['exit_code'] != 0
                or comparison['status'] != 'ORIGINAL_RECURRENT_REPLAY_EQUAL'
                or not comparison['all_native_datasets_and_tables_equal']):
            raise ValueError('V1_RECURRENT_PASSED_SOURCE_CANARY_REQUIRED')
        reference = Path(proof['reference'])
        original = json.loads((reference / 'report.json').read_text())
        if canary['rng'] != original['rng']:
            raise ValueError('V1_RECURRENT_CANARY_RNG_DIFFERS')
        package = work / 'Libs/population_dynamics'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        code_reference = os.environ.get('NRNSIM_V1_RECURRENT_CODE_REFERENCE')
        code_root = Path(code_reference) if code_reference else repo
        if not code_root.is_absolute():
            code_root = repo / code_root
        for name in modules:
            source = (repo if name == 'resource_probe' else code_root) / 'Libs/population_dynamics' / (name + '.py')
            if name != 'resource_probe' and digest(source) != digest(qualification / 'Libs/population_dynamics' / (name + '.py')):
                raise ValueError('V1_RECURRENT_IMPLEMENTATION_CHANGED_SINCE_CANARY:' + name)
            shutil.copy2(source, package / source.name)
        if code_reference:
            receipt['recurrent_code_reference'] = str(code_root)
        shutil.copy2(launcher, work / 'launcher.sbatch')
        config = yaml.safe_load(config_path.read_text())
        # Independent circuits use distinct seeds. Bind the complete source
        # recipe while leaving the approved realization identity variable.
        realization_keys = {'builder_seed', 'glif_assignment_seed', 'benchmark_circuit_candidate',
                            'source_root', 'bmtk_wheel'}
        if ({key: value for key, value in config.items() if key not in realization_keys}
                != {key: value for key, value in canary['original_config'].items()
                    if key not in realization_keys}
                or {key: value for key, value in config['bmtk_wheel'].items() if key != 'path'}
                != {key: value for key, value in canary['original_config']['bmtk_wheel'].items()
                    if key != 'path'}):
            raise ValueError('V1_RECURRENT_RECIPE_CHANGED_SINCE_CANARY')
        source_root = Path(config['source_root'])
        if not source_root.is_absolute():
            source_root = repo / source_root
        for name in source_names:
            for leaf in ('source.json', 'source_asset.txt'):
                target = work / 'sources' / name / leaf
                target.parent.mkdir(parents=True, exist_ok=True)
                source = source_root / name / leaf
                if digest(source) != digest(qualification / 'sources' / name / leaf):
                    raise ValueError('V1_RECURRENT_SOURCE_CHANGED_SINCE_CANARY:' + name + '/' + leaf)
                shutil.copy2(source, target)
        wheel = Path(config['bmtk_wheel']['path'])
        if not wheel.is_absolute():
            wheel = repo / wheel
        if digest(wheel) != config['bmtk_wheel']['sha256']:
            raise ValueError('V1_RECURRENT_PINNED_WHEEL_REQUIRED')
        shutil.copy2(wheel, work / 'bmtk.whl')
        shutil.copy2(config_path, work / 'original_config.yaml')
        resolved = dict(config, source_root=str(work / 'sources'),
            bmtk_wheel=dict(config['bmtk_wheel'], path=str(work / 'bmtk.whl')),
            original_config=config, original_config_sha256=digest(config_path))
        (work / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
        env = dict(os.environ, PYTHONPATH=str(work / 'Libs'), TMPDIR=temporary)
        receipt['canary_reference'] = str(qualification)
        receipt['qualified_canary_report_sha256'] = digest(qualification / 'canary/report.json')
        phases = [('resources', [sys.executable, '-B', '-m', 'population_dynamics.resource_probe',
                               '--output', str(work / 'resources.json'), '--required-memory-gib', '64' if mode == 'full' else '1'])]
        for label in ('primary', 'replay'):
            phases.append((label, [sys.executable, '-B', '-m', 'population_dynamics.v1_recurrent_runtime',
                '--config', str(work / 'resolved_config.yaml'), '--mode', mode, '--output', str(work / label)]))
        phases.append(('compare', [sys.executable, '-B', '-m', 'population_dynamics.v1_circuit_build',
            'compare', '--left', str(work / 'primary'), '--right', str(work / 'replay'), '--output', str(work / 'comparison')]))
        for label, command in phases:
            receipt['commands'].append(command)
            print(json.dumps(dict(job_id=job, phase=label, status='STARTED')), flush=True)
            with (work / (label + '.log')).open('w') as log:
                code = subprocess.run(command, env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
            print(json.dumps(dict(job_id=job, phase=label, exit_code=code)), flush=True)
            if code:
                break
        if code == 0:
            primary = json.loads((work / 'primary/report.json').read_text())
            replay = json.loads((work / 'replay/report.json').read_text())
            if primary['rng'] != replay['rng']:
                raise ValueError('V1_FULL_RECURRENT_REPLAY_RNG_DIFFERS')
            receipt.update(mode=primary['mode'],
                full_recurrent_realizations_generated=int(primary['full_recurrent_realization']))
        receipt.update(status='PASS_SOURCE_RECURRENT_BUILD_REPEATABILITY' if code == 0 else 'FAILED_SOURCE_RECURRENT_BUILD')
    except BaseException as error:
        code = code or 1
        receipt.update(status='FAILED_SOURCE_RECURRENT_BUILD', error_type=type(error).__name__, error=str(error))
        (work / 'failure.txt').write_text(traceback.format_exc())
    finally:
        receipt['child_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        destination.mkdir(parents=True, exist_ok=False)
        files = []
        for path in sorted(work.rglob('*')):
            relative = path.relative_to(work)
            if not path.is_file() or '__pycache__' in relative.parts or relative.name == 'bmtk.whl':
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            value = digest(path)
            if digest(target) != value:
                raise ValueError('V1_RECURRENT_DELIVERY_BYTES_DIFFER:' + str(relative))
            files.append(dict(path=str(relative), bytes=target.stat().st_size, sha256=value))
        receipt.update(files=files, exit_code=code, recorded_utc=datetime.now(timezone.utc).isoformat())
        (destination / 'delivery.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({key: value for key, value in receipt.items() if key not in ('files', 'commands')}), flush=True)
raise SystemExit(code)
PY

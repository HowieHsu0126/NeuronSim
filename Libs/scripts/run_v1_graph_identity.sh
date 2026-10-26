#!/bin/bash
# Validate the acquired source graph; no simulation or production preprocessing.
#SBATCH --job-name=NrnSim_v1_graph
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_graph-%j.log
set -euo pipefail
[[ $# == 3 ]] || { printf 'Expected repo acquired_run software_validation_archive\n'; exit 2; }
[[ ( ${SLURMD_NODENAME:-} == cpu04 || ${SLURMD_NODENAME:-} == cpu05 ) && -z ${SLURM_JOB_GPUS:-} ]] || exit 2
export NRNSIM_REPO=$1 NRNSIM_V1_ACQUIRED_RUN=$2 NRNSIM_V1_GRAPH_TESTS=$3
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
interpreter=${NRNSIM_V1_PYTHON:?Set NRNSIM_V1_PYTHON to the validated compute interpreter}
srun --ntasks=1 "$interpreter" -u - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, subprocess, sys, tempfile, traceback

repo = Path(os.environ['NRNSIM_REPO'])
acquired = Path(os.environ['NRNSIM_V1_ACQUIRED_RUN'])
tests = Path(os.environ['NRNSIM_V1_GRAPH_TESTS'])
intake = repo / 'Output/neural_population_dynamics/multicircuit/v1_asset_intake'
route = intake / 'original_graph_route'
work = Path(tempfile.mkdtemp(prefix='neuronsim-v1-graph-' + os.environ['SLURM_JOB_ID'] + '.'))
archive = route / 'graph_identity' / ('run_' + os.environ['SLURM_JOB_ID'])
report = dict(status='PREPARING', job_id=os.environ['SLURM_JOB_ID'], runtime=str(work),
              source_acquisition=str(acquired), source_scientific_fidelity_qualified=False,
              production_episodes=0, started_utc=datetime.now(timezone.utc).isoformat())
code = 1
try:
    (work / 'Libs/population_dynamics').mkdir(parents=True)
    (work / 'Libs/population_dynamics/__init__.py').write_bytes(b'')
    (work / 'scratch').mkdir()
    (work / 'phases').mkdir()
    names = ['v1_graph_identity', 'test_v1_graph_identity', 'provenance', 'resource_probe']
    for name in names:
        shutil.copy2(repo / 'Libs/population_dynamics' / (name + '.py'),
                     work / 'Libs/population_dynamics' / (name + '.py'))
    sys.path.insert(0, str(work / 'Libs'))
    from population_dynamics.provenance import sha256_file, versions
    from population_dynamics.resource_probe import probe
    if probe(work / 'resource.json', 32)['status'] != 'PASS':
        raise RuntimeError('V1_GRAPH_RESOURCE_BLOCKED')
    validation = json.loads((tests / 'delivery.json').read_text())
    if validation['status'] != 'PASS' or not validation['verified']:
        raise ValueError('V1_GRAPH_SOFTWARE_NOT_VALIDATED')
    rows = {row['path']: row for row in validation['files']}
    for name in names:
        relative = 'Libs/population_dynamics/' + name + '.py'
        if sha256_file(work / relative) != rows[relative]['sha256']:
            raise ValueError('V1_GRAPH_TESTED_SOURCE_CHANGED:' + name)
    report['software_validation'] = dict(archive=str(tests),
        delivery_sha256=sha256_file(tests / 'delivery.json'), unchanged_modules=names)
    delivery = json.loads((acquired / 'delivery.json').read_text())
    if (delivery['status'] != 'ARCHIVED_V1_SOURCE_ASSET' or delivery['exit_code'] != 0
            or not delivery['asset_body_acquired'] or delivery['retained_local_payloads']):
        raise ValueError('V1_COMPLETE_ACQUISITION_REQUIRED')
    source_row = next(row for row in delivery['files'] if row['path'] == 'asset/source.json')
    if sha256_file(acquired / source_row['path']) != source_row['sha256']:
        raise ValueError('V1_ACQUISITION_RECEIPT_CHANGED')
    source = json.loads((acquired / source_row['path']).read_text())
    if source['status'] != 'SOURCE_ASSET_ACQUIRED':
        raise ValueError('V1_COMPLETE_SOURCE_REQUIRED')
    shutil.copy2(repo / 'Libs/scripts/run_v1_graph_identity.sh', work / 'launcher.sh')
    shutil.copy2(intake.parent / 'ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py', work / 'run_guarded.py')
    inputs = {
        'download.json': route / 'zenodo_download_receipt.json',
        'native_nodes.h5': intake / 'node_metadata_complete/v1_nodes.h5',
        'native_types.csv': intake / 'node_types/source_asset.txt',
        'node_validation.json': intake / 'node_validation.json',
        'load_sparse.py': route / 'author_graph_loader/load_sparse.py',
        'loader_inspection.json': route / 'author_supporting_record_19965055.json',
    }
    for name, path in inputs.items():
        shutil.copy2(path, work / name)
    env = dict(os.environ, PYTHONPATH=str(work / 'Libs'), TMPDIR=str(work / 'scratch'))
    copy = [sys.executable, '-c', 'import shutil,sys; shutil.copy2(sys.argv[1],sys.argv[2])',
            str(acquired / 'asset/GLIF_network.zip'), str(work / 'GLIF_network.zip')]
    result = subprocess.run([sys.executable, str(work / 'run_guarded.py'), 'stage_archive', *copy],
                            cwd=work, env=env)
    if result.returncode:
        raise RuntimeError('V1_ARCHIVE_STAGING_FAILED')
    command = [sys.executable, '-m', 'population_dynamics.v1_graph_identity',
        '--archive', str(work / 'GLIF_network.zip'), '--download-receipt', str(work / 'download.json'),
        '--reference-nodes', str(work / 'native_nodes.h5'), '--reference-types', str(work / 'native_types.csv'),
        '--reference-report', str(work / 'node_validation.json'), '--loader', str(work / 'load_sparse.py'),
        '--loader-inspection', str(work / 'loader_inspection.json'), '--output', str(work / 'graph_identity.json')]
    report['command'] = command
    result = subprocess.run([sys.executable, str(work / 'run_guarded.py'), 'inspect', *command],
                            cwd=work, env=env)
    phase = json.loads((work / 'phases/inspect/receipt.json').read_text())
    report.update(inspector_exit_code=phase['exit_code'], guard_exit_code=result.returncode,
                  versions=versions())
    if (work / 'graph_identity.json').is_file():
        graph = json.loads((work / 'graph_identity.json').read_text())
        report.update(status='INSPECTED' if phase['exit_code'] == 0 else 'INSPECTED_REQUIRES_RECONCILIATION',
                      cortical_neurons=graph['cortical_neurons'], raw_synapses=graph['raw_synapses'],
                      topology=graph['topology'], loader_delay_alignment=graph['loader_delay_alignment'])
        code = phase['exit_code'] if phase['exit_code'] is not None else 1
    else:
        raise RuntimeError('V1_GRAPH_INSPECTION_DID_NOT_COMPLETE')
except Exception as exc:
    report.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
    (work / 'failure.txt').write_text(traceback.format_exc())
finally:
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (work / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
    archive.mkdir(parents=True, exist_ok=False)
    files = []
    from hashlib import sha256
    # Large input replicas and extracted pickle are scratch; the acquired archive
    # remains the immutable source. Retain code, metadata and inspection evidence.
    for path in sorted(work.rglob('*')):
        relative = path.relative_to(work)
        if (not path.is_file() or path.is_symlink() or relative.parts[0] == 'scratch'
                or relative.name == 'GLIF_network.zip' or '__pycache__' in relative.parts):
            continue
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = sha256(path.read_bytes()).hexdigest()
        if sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('V1_GRAPH_OUTPUT_COPY_CHANGED:' + str(relative))
        files.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=digest))
    (archive / 'delivery.json').write_text(json.dumps(dict(status=report['status'],
        job_id=os.environ['SLURM_JOB_ID'], exit_code=code, source_acquisition=str(acquired),
        files=files, copied_files_verified=True, production_episodes=0), indent=2) + '\n')
    shutil.rmtree(work)
    print(json.dumps(report, indent=2), flush=True)
sys.exit(code)
PY

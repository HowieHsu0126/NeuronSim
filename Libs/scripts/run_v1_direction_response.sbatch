#!/bin/bash
#SBATCH --job-name=NrnSim_V1DirectionReadout
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
#SBATCH --output=/tmp/NrnSim_V1DirectionReadout-%j.log
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${NRNSIM_READOUT_PYTHON:?Set the validated readout interpreter}" -B - "$@" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

import yaml

repo = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
submission_path, guard_reference = map(Path, sys.argv[3:5])
validation = Path(sys.argv[5]) if len(sys.argv) == 6 else None
if mode not in ('validation', 'all') or (mode == 'all' and validation is None):
    raise ValueError('validation or all with accepted software reference required')
if os.environ.get('SLURMD_NODENAME') != 'cpu04' or os.environ.get('SLURM_JOB_GPUS'):
    raise ValueError('CPU04 without a GPU is required')
job = os.environ['SLURM_JOB_ID']
family = repo/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/direction_coverage'
out = family/('validation' if mode == 'validation' else 'readout')/('run_'+job)
out.mkdir(parents=True, exist_ok=False)
receipt = dict(job_id=job, mode=mode, status='PREPARING', exit_code=1,
    started_utc=datetime.now(timezone.utc).isoformat(), submission=str(submission_path),
    guard_reference=str(guard_reference), scientific_qualification=False, production_episodes=0)
protocol_name = 'Libs/configs/neural_population_v1_direction_development.yaml'
visual_name = 'Libs/configs/neural_population_v1_constructed_visual_development.yaml'
launcher_name = 'Libs/scripts/run_v1_direction_response.sbatch'
modules = ('circuit_campaign', 'circuit_data', 'config', 'population_field', 'provenance',
    'test_v1_direction_response', 'v1_circuit_stimulus', 'v1_clock_comparison',
    'v1_configured_inputs', 'v1_direction_response', 'v1_lgn_control_causality',
    'v1_lgn_runtime', 'v1_lgn_source', 'v1_lgn_spikes', 'v1_nest_input_replay',
    'v1_nest_repeatability', 'v1_nest_time_semantics', 'v1_source_dataset', 'v1_visual_controls')
code_names = ['Libs/population_dynamics/'+name+'.py' for name in modules]
code_names += [protocol_name, visual_name, launcher_name]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest(source):
    value = json.loads((source/'delivery.json').read_text())
    rows = {row['path']: row for row in value['files']}
    if value.get('exit_code') != 0 or len(rows) != len(value['files']):
        raise ValueError('Complete unique delivery required: '+str(source))
    return value, rows


def copy_bound(source, destination, name, rows):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid declared input path: '+name)
    target = destination/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/relative, target)
    row = rows[name]
    if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
        raise ValueError('Changed declared input: '+str(source/relative))
    return target


try:
    _, guard_rows = manifest(guard_reference)
    software_rows = None
    if mode == 'all':
        vd, software_rows = manifest(validation)
        if vd['status'] != 'PASS_DIRECTION_SOFTWARE_AND_REFERENCE_INTAKE':
            raise ValueError('Accepted direction software and real reference intake required')
        for name in (protocol_name, visual_name):
            if sha(repo/name) != software_rows[name]['sha256']:
                raise ValueError('Registered configuration changed after validation: '+name)
        receipt['validation_reference'] = str(validation)
        receipt['validation_delivery_sha256'] = sha(validation/'delivery.json')
    with tempfile.TemporaryDirectory(prefix='neuronsim-v1-directions-'+job+'.', dir='/tmp') as temp:
        work = Path(temp)
        for name in ('scratch', 'phases', 'sources', 'input'):
            (work/name).mkdir()
        for name in ('run_guarded.py', 'Libs/population_dynamics/resource_probe.py'):
            copy_bound(guard_reference, work, name, guard_rows)
        (work/'Libs/population_dynamics/__init__.py').write_text('')
        for name in code_names:
            if mode == 'all':
                copy_bound(validation, work, name, software_rows)
            else:
                target = work/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo/name, target)
        # Preserve the exact implementation even if an import or staging check fails.
        shutil.copytree(work/'Libs', out/'Libs')
        shutil.copyfile(work/'run_guarded.py', out/'run_guarded.py')
        sys.path.insert(0, str(work/'Libs'))
        from population_dynamics.resource_probe import probe
        from population_dynamics.v1_clock_comparison import required_files
        from population_dynamics.v1_direction_response import _protocol, SPATIAL_REFERENCE_FILES
        health = probe(work/'staging_preflight.json', required_memory_gib=4)
        shutil.copyfile(work/'staging_preflight.json', out/'staging_preflight.json')
        if health['status'] != 'PASS':
            raise RuntimeError('Readout staging resource gate failed')
        shutil.copyfile(submission_path, work/'input/submission.json')
        protocol, _, _, jobs = _protocol(work/protocol_name, work/'input/submission.json')
        reference = repo/protocol['references']['reused_native']
        origins = {315: reference}
        if mode == 'all':
            origins.update({degree: reference.parent/('run_'+job_id)
                            for degree, job_id in jobs.items() if degree != 315})
        # Inspect only declared manifests/reports, never traverse source archives.
        sources, total_bytes = {}, 0
        for degree, source in origins.items():
            delivery, rows = manifest(source)
            report_bytes = (source/'primary/report.json').read_bytes()
            binding = rows['primary/report.json']
            if len(report_bytes) != binding['bytes'] or hashlib.sha256(report_bytes).hexdigest() != binding['sha256']:
                raise ValueError('Native primary report changed: '+str(source))
            names = required_files(json.loads(report_bytes))
            total_bytes += sum(rows[name]['bytes'] for name in names if name != 'delivery.json')
            sources[degree] = (source, rows, names)
        spatial_source = repo/protocol['references']['reused_observation']
        _, spatial_rows = manifest(spatial_source)
        total_bytes += sum(spatial_rows[name]['bytes'] for name in SPATIAL_REFERENCE_FILES if name != 'delivery.json')
        if shutil.disk_usage('/tmp').free < 20*1024**3 + 2*total_bytes:
            raise RuntimeError('Insufficient scratch with unchanged 20 GiB root reserve')
        archives = {}
        for degree, (source, rows, names) in sources.items():
            target = work/'sources'/str(degree)
            target.mkdir()
            shutil.copyfile(source/'delivery.json', target/'delivery.json')
            for name in names:
                if name != 'delivery.json':
                    copy_bound(source, target, name, rows)
            archives[str(degree)] = str(target)
        spatial_target = work/protocol['references']['reused_observation']
        spatial_target.mkdir(parents=True)
        shutil.copyfile(spatial_source/'delivery.json', spatial_target/'delivery.json')
        for name in SPATIAL_REFERENCE_FILES[1:]:
            copy_bound(spatial_source, spatial_target, name, spatial_rows)
        (work/'input/archives.json').write_text(json.dumps(archives, indent=2)+'\n')
        receipt.update(native_sources={str(k): str(v) for k, v in origins.items()},
            declared_native_files=sum(len(value[2]) for value in sources.values()),
            staged_bytes=total_bytes, submission_sha256=sha(work/'input/submission.json'),
            guard_delivery_sha256=sha(guard_reference/'delivery.json'))
        env = dict(os.environ, PYTHONPATH=str(work/'Libs'), TMPDIR=str(work/'scratch'),
            MPLCONFIGDIR=str(work/'mpl'), XDG_CACHE_HOME=str(work/'cache'), NUMBA_CACHE_DIR=str(work/'numba'))
        phases = []
        if mode == 'validation':
            phases.append(('tests', [sys.executable, '-B', '-m', 'pytest',
                str(work/'Libs/population_dynamics/test_v1_direction_response.py'), '-q', '-p', 'no:cacheprovider',
                '--basetemp='+str(work/'scratch/pytest'), '--junitxml='+str(work/'tests.xml')]))
        analysis_dir = 'reference_intake' if mode == 'validation' else 'direction_response'
        command = [sys.executable, '-B', '-m', 'population_dynamics.v1_direction_response',
            '--protocol', str(work/protocol_name), '--submission', str(work/'input/submission.json'),
            '--archives-json', str(work/'input/archives.json'), '--output', str(work/analysis_dir)]
        if mode == 'validation':
            command.append('--intake-only')
        phases.append((analysis_dir, command))
        try:
            receipt['commands'] = []
            for phase, command in phases:
                guarded = [sys.executable, '-B', str(work/'run_guarded.py'), phase, *command]
                result = subprocess.run(guarded, cwd=work, env=env, capture_output=True, text=True)
                (work/(phase+'_launcher.log')).write_text(result.stdout+result.stderr)
                receipt['commands'].append(dict(phase=phase, command=guarded, exit_code=result.returncode))
                if result.returncode:
                    raise RuntimeError('Direction readout phase failed: '+phase)
            result = json.loads((work/analysis_dir/'report.json').read_text())
            if mode == 'validation':
                suites = list(ET.parse(work/'tests.xml').getroot().iter('testsuite'))
                counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
                if (counts['tests'] == 0 or any(counts[k] for k in ('failures', 'errors', 'skipped'))
                        or result['status'] != 'V1_DIRECTION_INTAKE_ONLY'
                        or result['complete_eight_direction_coverage'] or result['admitted_conditions'] != 1):
                    raise ValueError('Complete software checks and real reference intake required')
                receipt['junit'] = counts
                status = 'PASS_DIRECTION_SOFTWARE_AND_REFERENCE_INTAKE'
            else:
                if (result['status'] != 'V1_DIRECTION_RESPONSE_COMPLETE_DEVELOPMENT'
                        or not result['complete_eight_direction_coverage'] or result['admitted_conditions'] != 8):
                    raise ValueError('Complete eight-direction readout required')
                status = 'COMPLETE_DIRECTION_RESPONSE_DEVELOPMENT'
            receipt.update(status=status, exit_code=0)
        finally:
            for name in ['input', 'phases', analysis_dir]:
                if (work/name).exists():
                    shutil.copytree(work/name, out/name)
            for name in ['tests.xml', 'tests_launcher.log', analysis_dir+'_launcher.log']:
                if (work/name).exists():
                    shutil.copyfile(work/name, out/name)
except Exception:
    receipt.update(status='FAILED_DIRECTION_RESPONSE_EXECUTION', exit_code=1, error=traceback.format_exc())
finally:
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out/'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    files = [dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path))
             for path in out.rglob('*') if path.is_file()]
    (out/'delivery.json').write_text(json.dumps(dict(status=receipt['status'], job_id=job,
        exit_code=receipt['exit_code'], files=files), indent=2)+'\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(receipt['exit_code'])
PY

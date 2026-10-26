#!/bin/bash
#SBATCH --job-name=NrnSim_ClockBankSummary
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_ClockBankSummary-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PFC_PYTHON:?PFC_PYTHON is required}" -B - "$@" <<'PY'
import hashlib, json, os, shutil, subprocess, sys, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path

repo, source, pilot, validation = map(Path, sys.argv[1:5])
job = os.environ['SLURM_JOB_ID']
out = repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/kernel_parameterization/recurrent_gain/coupled_clock_replicates/summary'/('run_'+job)
out.mkdir(parents=True, exist_ok=False)
receipt = dict(job_id=job, source=str(source), pilot_summary=str(pilot), validation=str(validation),
               started_utc=datetime.now(timezone.utc).isoformat(), status='PREPARING', exit_code=None)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked(root, name, delivery):
    entry = next(row for row in delivery['files'] if row['path'] == name)
    path = root/name
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError('delivered file differs: '+name)
    return path

try:
    vd = json.loads((validation/'delivery.json').read_text())
    if vd['status'] != 'PASS_SOFTWARE_AND_ACCEPTED_PILOT_INTAKE' or vd['exit_code'] != 0:
        raise ValueError('accepted summary software required')
    vr = json.loads(checked(validation, 'report.json', vd).read_text())
    implementation = 'Libs/scripts/summarize_pfc_coupled_clock.py'
    protocol = 'Libs/configs/neural_population_pfc_recurrent_gain_coupled_clock_replicates.yaml'
    if sha(repo/protocol) != vr['implementation_sha256'][protocol]:
        raise ValueError('registered source protocol changed after validation')
    pd = json.loads((pilot/'delivery.json').read_text())
    sd = json.loads((source/'delivery.json').read_text())
    report = json.loads(checked(source, 'analysis/report.json', sd).read_text())
    if report['complete_source_episodes_consumed'] != 522 or len(report['blocks']) != 58:
        raise ValueError('complete remaining29 reader output required')
    names = ['report.json', 'analysis/report.json']
    block_files = ['source_response_comparison.csv', 'paired_response_comparison.csv',
                   'field_clock_discrepancies.csv', 'shared_input_checks.csv',
                   'semantic_phase_readouts.csv', 'count_cdf_comparison.csv',
                   'unit_count_clock_differences.npz']
    for block in report['blocks']:
        names.extend(f"analysis/{block['cohort']}/{block['source_rng_seed']}/{name}" for name in block_files)
    entries = {row['path']: row for row in sd['files']}
    stage_bytes = sum(entries[name]['bytes'] for name in names)
    if shutil.disk_usage('/tmp').free < 2*stage_bytes + 20*1024**3:
        raise ValueError('insufficient local scratch with unchanged root reserve')
    with tempfile.TemporaryDirectory(prefix='NrnSim_clock_bank_summary_', dir='/tmp') as temp:
        work = Path(temp)
        for name in ['run_guarded.py', 'Libs/population_dynamics/__init__.py',
                     'Libs/population_dynamics/resource_probe.py']:
            target = work/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked(pilot, name, pd), target)
        sys.path.insert(0, str(work/'Libs'))
        from population_dynamics.resource_probe import probe
        if probe(work/'staging_preflight.json', required_memory_gib=1)['status'] != 'PASS':
            raise RuntimeError('staging resource preflight failed')
        for folder in ('scratch', 'phases', 'source', 'pilot/summary'):
            (work/folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(validation, implementation, vd), work/'summarize_pfc_coupled_clock.py')
        shutil.copyfile(repo/protocol, work/'protocol.yaml')
        shutil.copyfile(source/'delivery.json', work/'source/delivery.json')
        for name in names:
            target = work/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
        shutil.copyfile(pilot/'delivery.json', work/'pilot/delivery.json')
        for row in pd['files']:
            if row['path'].startswith('summary/'):
                shutil.copyfile(pilot/row['path'], work/'pilot'/row['path'])
        command = [sys.executable, '-B', str(work/'run_guarded.py'), 'summarize',
                   sys.executable, '-B', str(work/'summarize_pfc_coupled_clock.py'),
                   '--source', str(work/'source'), '--output', str(work/'summary'),
                   '--protocol', str(work/'protocol.yaml'), '--pilot-summary', str(work/'pilot')]
        receipt.update(command=command, declared_source_files=len(names), staged_source_bytes=stage_bytes,
                       validation_delivery_sha256=sha(validation/'delivery.json'),
                       source_delivery_sha256=sha(source/'delivery.json'),
                       pilot_delivery_sha256=sha(pilot/'delivery.json'))
        run = subprocess.run(command, cwd=work, capture_output=True, text=True)
        (out/'command.log').write_text(run.stdout+run.stderr)
        for name in ('summary', 'phases'):
            if (work/name).exists():
                shutil.copytree(work/name, out/name)
        for name in ('staging_preflight.json', 'summarize_pfc_coupled_clock.py', 'protocol.yaml', 'run_guarded.py'):
            shutil.copyfile(work/name, out/name)
        receipt.update(status='COMPLETE' if run.returncode == 0 else 'FAILED', exit_code=run.returncode)
except Exception:
    receipt.update(status='FAILED', exit_code=1, error=traceback.format_exc())
finally:
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out/'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    files = [dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path))
             for path in out.rglob('*') if path.is_file()]
    (out/'delivery.json').write_text(json.dumps(dict(status=receipt['status'], job_id=job,
        exit_code=receipt['exit_code'], source_job=source.name, files=files), indent=2)+'\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(receipt['exit_code'])
PY

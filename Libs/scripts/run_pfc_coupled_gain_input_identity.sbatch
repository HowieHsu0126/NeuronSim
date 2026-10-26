#!/bin/bash
#SBATCH --job-name=NrnSim_GainInputIdentity
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-gain-input-identity-%j.out
#SBATCH --error=/tmp/neuronsim-gain-input-identity-%j.err
set -euo pipefail
: "${PROJECT_PREFIX:?PROJECT_PREFIX_REQUIRED}"
: "${NEURONSIM_PROJECT_ROOT:?NEURONSIM_PROJECT_ROOT_REQUIRED}"
: "${NRNSIM_PYTHON:?NRNSIM_PYTHON_REQUIRED}"
: "${NRNSIM_INPUT_IDENTITY_OUTPUT_ROOT:?OUTPUT_ROOT_REQUIRED}"
[[ "$PROJECT_PREFIX" == "NrnSim" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
unset PYTHONPATH
export NRNSIM_INPUT_IDENTITY_BATCH="$0"
"$NRNSIM_PYTHON" -B - "$@" <<'PY_IDENTITY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

project=Path(os.environ['NEURONSIM_PROJECT_ROOT']).resolve()
job=os.environ['SLURM_JOB_ID']
destination=Path(os.environ['NRNSIM_INPUT_IDENTITY_OUTPUT_ROOT']).resolve()/('run_'+job)
if destination.exists():
    raise FileExistsError('IMMUTABLE_JOB_OUTPUT_ALREADY_EXISTS:'+str(destination))
module='pfc_coupled_gain_input_identity.py'
test='test_pfc_coupled_gain_input_identity.py'
commands=[]
status='FAILED_GAIN_INPUT_IDENTITY_VALIDATION'
exit_code=1
started=time.monotonic()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='neuronsim-gain-input-identity-'+job+'.',dir='/tmp') as temporary:
    work=Path(temporary)
    snapshots=[]
    try:
        if shutil.disk_usage(work).free < 1024**3:
            raise RuntimeError('ONE_GIB_NODE_LOCAL_SCRATCH_REQUIRED')
        for name in (module,test):
            source=project/'Libs/population_dynamics'/name
            shutil.copyfile(source,work/name)
            snapshots.append(dict(path=name,source=str(source),sha256=sha(work/name)))
        shutil.copyfile(os.environ['NRNSIM_INPUT_IDENTITY_BATCH'],work/'launcher.sbatch')
        snapshots.append(dict(path='launcher.sbatch',sha256=sha(work/'launcher.sbatch')))
        def execute(label,argv):
            begin=time.monotonic()
            with (work/(label+'.stdout')).open('w') as stdout,(work/(label+'.stderr')).open('w') as stderr:
                result=subprocess.run(argv,cwd=work,stdout=stdout,stderr=stderr,check=False)
            commands.append(dict(phase=label,argv=argv,exit_code=result.returncode,wall_seconds=time.monotonic()-begin))
            print(json.dumps(commands[-1]),flush=True)
            if result.returncode:
                raise RuntimeError(label+'_FAILED:'+str(result.returncode))
        execute('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(work/test),'--basetemp',str(work/'pytest_tmp'),'--junitxml',str(work/'tests.xml')])
        execute('identity',[sys.executable,'-B',str(work/module),*sys.argv[1:],'--output',str(work/'analysis')])
        result=json.loads((work/'analysis/report.json').read_text())
        if result['status']!='PASS_STORED_COUPLED_GAIN_INPUT_IDENTITY' or result['passed_pairs']!=6:
            raise RuntimeError('SIX_COMPLETE_STORED_INPUT_PAIRS_REQUIRED')
        status='COMPLETE_GAIN_INPUT_IDENTITY_VALIDATION'
        exit_code=0
    except Exception as error:
        (work/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)),indent=2)+'\n')
    report=dict(status=status,exit_code=exit_code,job_id=job,node=os.environ.get('SLURMD_NODENAME'),commands=commands,snapshots=snapshots,
        source_simulations=0,new_rng_draws=0,production_episodes=0,scientific_fidelity_qualified=False,
        wall_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat(),python=sys.version)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    names=[module,test,'launcher.sbatch','tests.stdout','tests.stderr','tests.xml','identity.stdout','identity.stderr','analysis/report.json','report.json','failure.json']
    destination.mkdir(parents=True,exist_ok=False)
    delivered=[]
    for name in names:
        source=work/name
        if not source.is_file():
            continue
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        source_sha=sha(source)
        shutil.copyfile(source,target)
        if sha(target)!=source_sha or target.stat().st_size!=source.stat().st_size:
            raise RuntimeError('DELIVERY_COPY_CHANGED:'+name)
        delivered.append(dict(path=name,bytes=source.stat().st_size,sha256=source_sha))
    (destination/'delivery.json').write_text(json.dumps(dict(status=status,job_id=job,verified=True,files=delivered),indent=2)+'\n')
    print(json.dumps(dict(status=status,exit_code=exit_code,archive=str(destination))),flush=True)
raise SystemExit(exit_code)
PY_IDENTITY

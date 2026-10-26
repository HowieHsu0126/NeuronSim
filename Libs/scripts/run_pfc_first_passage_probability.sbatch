#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_FirstPassageProbability
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-first-passage-probability-%j.out
#SBATCH --error=/tmp/neuronsim-first-passage-probability-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_PROBABILITY_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_PROBABILITY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import yaml

assert socket.gethostname().split('.')[0] == 'cpu04'
assert not os.environ.get('SLURM_JOB_GPUS')
repo=Path(os.environ['NEURONSIM_PROJECT_ROOT'])
assert (repo/'PROJECT_PREFIX').read_text().strip()=='NrnSim'
job=os.environ['SLURM_JOB_ID']
method=os.environ.get('NEURONSIM_PASSAGE_SAMPLER','plain')
assert method in ('plain','importance')
stem='pfc_first_passage_probability' if method=='plain' else 'pfc_first_passage_importance'
status_name='PFC_FIRST_PASSAGE_PROBABILITY' if method=='plain' else 'PFC_FIRST_PASSAGE_IMPORTANCE'
archive=repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement'/stem.removeprefix('pfc_')/('run_'+job)
assert not archive.exists()
work=Path(tempfile.mkdtemp(prefix='neuronsim-first-passage-probability-'+job+'.',dir='/tmp'))
(work/'scratch').mkdir()
env=dict(os.environ,PYTHONPATH=str(work),TMPDIR=str(work/'scratch'))
started=time.perf_counter();commands=[];inputs=[];error=None;delivered=False;exit_code=1
status='FAILED_'+status_name
config=Path('Libs/configs')/('neural_population_'+stem+'.yaml')
module=Path('Libs/population_dynamics')/(stem+'.py')
test=Path('Libs/population_dynamics')/('test_'+stem+'.py')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(source,relative,expected=None):
    target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    actual=digest(target)
    if expected is not None and actual!=expected:raise ValueError('FIRST_PASSAGE_PROBABILITY_INPUT_CHANGED:'+str(relative))
    inputs.append(dict(source=str(source),path=str(relative),sha256=actual))

def phase(name,args):
    begin=time.perf_counter()
    with (work/(name+'.stdout')).open('w') as out,(work/(name+'.stderr')).open('w') as err:
        result=subprocess.run(args,cwd=work,env=env,stdout=out,stderr=err)
    commands.append(dict(phase=name,command=args,exit_code=result.returncode,wall_seconds=time.perf_counter()-begin))
    print(json.dumps(commands[-1]),flush=True)
    if result.returncode:raise RuntimeError('FIRST_PASSAGE_PROBABILITY_'+name.upper()+'_FAILED:'+str(result.returncode))

try:
    copy(repo/config,config,os.environ['NEURONSIM_PROBABILITY_CONFIG_SHA256'])
    settings=yaml.safe_load((work/config).read_text())
    assert settings['schema']==('pfc_conditional_first_passage_probability_v1' if method=='plain' else 'pfc_conditional_first_passage_importance_v1')
    reference=repo/settings['reference']['archive']
    copy(reference/'delivery.json',Path('reference_delivery.json'),settings['reference']['delivery_sha256'])
    delivery=json.loads((work/'reference_delivery.json').read_text())
    assert delivery['verified'] and delivery['status']=='COMPLETE_PFC_FIRST_PASSAGE_REFERENCE_VALIDATION'
    for entry in delivery['files']:
        relative=Path(entry['path'])
        if relative.parts[0]=='Libs' and relative.suffix=='.py':
            copy(reference/relative,relative,entry['sha256'])
        elif entry['path'] in ('report.json','conditional_initial.json','reference_window.npz'):
            copy(reference/relative,Path('reference')/relative,entry['sha256'])
    report=json.loads((work/'reference/report.json').read_text())
    assert report['all_snapshot_digests_equal'] and report['source_prefix_events_equal'] and report['all_prefix_rng_state_equal']
    assert all(report['terminal_state_checks'].values())
    source_config=next(row for row in report['inputs'] if row['path']=='source_inputs/config.yaml')
    copy(Path(source_config['source']),Path('source_config.yaml'),source_config['sha256'])
    if method=='importance':
        kernel=repo/settings['kernel']['archive']
        copy(kernel/'delivery.json',Path('kernel_delivery.json'),settings['kernel']['delivery_sha256'])
        kd=json.loads((work/'kernel_delivery.json').read_text())
        assert kd['verified'] and kd['status']=='COMPLETE_PFC_FIRST_PASSAGE_PROBABILITY'
        entries={row['path']:row for row in kd['files']}
        relative=Path('Libs/population_dynamics/pfc_first_passage_probability.py')
        copy(kernel/relative,relative,entries[str(relative)]['sha256'])
        copy(kernel/'report.json',Path('kernel_report.json'),entries['report.json']['sha256'])
        kr=json.loads((work/'kernel_report.json').read_text())
        assert kr['exit_code']==0 and sum(int(row['tests']) for row in kr['test_suites'])==6
        commands.append(dict(phase='source_kernel_tests_reused',archive=str(kernel),tests=6,exit_code=0))
    for relative in (module,test):copy(repo/relative,relative)
    copy(Path(os.environ['NEURONSIM_PROBABILITY_LAUNCHER']),Path('launcher.sbatch'))
    sys.path.insert(0,str(work))
    from Libs.population_dynamics.resource_probe import probe
    health=probe(work/'resource_preflight.json',4)
    assert health['status']=='PASS',health['reasons']
    phase('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(test),
                   '--basetemp='+str(work/'test_tmp'),'--junitxml='+str(work/'tests.xml')])
    phase('sampling',[sys.executable,'-B','-m','Libs.population_dynamics.'+stem,
                      '--config',str(work/config),'--source-config',str(work/'source_config.yaml'),
                      '--reference',str(work/'reference'),'--output',str(work/'analysis')])
    status='COMPLETE_'+status_name;exit_code=0
except Exception as exc:
    error=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
finally:
    suites=[]
    if (work/'tests.xml').exists():suites=[dict(node.attrib) for node in ET.parse(work/'tests.xml').getroot().iter('testsuite')]
    result=dict(status=status,exit_code=exit_code,error=error,job_id=job,node=socket.gethostname(),
        inputs=inputs,commands=commands,test_suites=suites,sampler_method=method,wall_seconds=time.perf_counter()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(),new_full_network_episodes=0,
        new_independent_source_blocks=0,population_history_uncertainty_established=False,
        confirmation_execution_released=False,production_episodes=0)
    (work/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        archive.mkdir(parents=True,exist_ok=False);files=[]
        for path in sorted(work.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(work)
            if relative.parts[0] in ('test_tmp','scratch') or '__pycache__' in relative.parts:continue
            target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            checksum=digest(path)
            assert digest(target)==checksum
            files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=checksum))
        (archive/'delivery.json').write_text(json.dumps(dict(status=status,verified=True,job_id=job,files=files),indent=2)+'\n')
        delivered=True;print(json.dumps(dict(status=status,archive=str(archive),exit_code=exit_code)),flush=True)
    finally:
        if delivered:shutil.rmtree(work)
        else:print('COPYBACK_FAILED_SCRATCH_PRESERVED:'+str(work),file=sys.stderr)
raise SystemExit(exit_code)
PY_PROBABILITY

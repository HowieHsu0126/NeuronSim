#!/bin/bash
#SBATCH --job-name=NrnSim_HistoryTailBound
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --output=/tmp/neuronsim-history-tail-bound-%j.out
#SBATCH --error=/tmp/neuronsim-history-tail-bound-%j.err
set -euo pipefail
bound_launch_tmp=$(mktemp -d /tmp/neuronsim-tail-launch.XXXXXX)
trap 'rm -rf "$bound_launch_tmp"' EXIT
export NEURONSIM_BOUND_CODE_INPUTS='{"Libs/population_dynamics/pfc_history_tail_bound.py": "3d324d4a223e8ed2fccc17ac172f948b96bf7891d389106f079e1495d8447c64", "Libs/population_dynamics/test_pfc_history_tail_bound.py": "68036362a5b0105ff3fc5067f54a4c39efe180a40e228827f15de067f01f8341", "Libs/population_dynamics/test_pfc_first_passage_importance.py": "b4f645ea459e12b1ba97dcdd778af78efb2595e17c6f38cf13c29b64295a0a74", "Libs/configs/neural_population_pfc_history_tail_bound.yaml": "c60b25521b80360b58ab6f5c0dcd8262e8a7c7a2c93d820e458ed00de97dc393", "Libs/population_dynamics/pfc_first_passage_importance.py": "0e07a61c94d19acdb6d03839815077cc6e4084c905786a42e23d3576f20adc11"}'
cat > "$bound_launch_tmp/validate.py" <<'PY_VALIDATION'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

repo = Path(os.environ['NEURONSIM_PROJECT_ROOT'])
base = Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement')
job = os.environ['SLURM_JOB_ID']
report = dict(status='RUNNING', job_id=job, started_utc=datetime.now(timezone.utc).isoformat(),
              inputs=[], commands=[], new_source_simulations=0, scientific_fidelity_qualified=False)
output = repo/base/'history_tail_bound'/('run_'+job)
output.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__,output/'validation.py')
started = time.perf_counter()
exit_code = 1
with tempfile.TemporaryDirectory(prefix='neuronsim-history-tail-bound-',dir='/tmp') as scratch:
    work = Path(scratch)
    def bound_copy(origin, target, expected):
        raw = origin.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected: raise ValueError('INPUT_BINDING:'+str(origin))
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        report['inputs'].append(dict(source=str(origin),target=str(target.relative_to(work)),sha256=digest,bytes=len(raw)))
    def command(args, name, expected=0):
        with (output/(name+'.stdout')).open('w') as out, (output/(name+'.stderr')).open('w') as err:
            result = subprocess.run(args,cwd=work,env={**os.environ,'PYTHONPATH':str(work),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''},stdout=out,stderr=err)
        report['commands'].append(dict(name=name,command=args,exit_code=result.returncode,expected_exit_code=expected))
        if result.returncode != expected: raise RuntimeError(name+':'+str(result.returncode))
    try:
        prepared = repo/base/'profile_history_readback/prepare/run_605110'
        raw = (prepared/'delivery.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == '8c61ae4021c20650f778240bffbaf7ee804ee5cc91b857e2f6e8a57256f0e7a0'
        delivery = json.loads(raw)
        for entry in delivery['files']:
            name = entry['path']
            if name.startswith('Libs/') and name.endswith('.py') and '/test_' not in name:
                bound_copy(prepared/name,work/name,entry['sha256'])
        pending = json.loads(os.environ['NEURONSIM_BOUND_CODE_INPUTS'])
        for name,digest in pending.items():
            bound_copy(repo/name,work/name,digest)
        command([sys.executable,'-m','Libs.population_dynamics.resource_probe','--output',str(output/'resource_preflight.json'),'--required-memory-gib','1'],'resources')
        command([sys.executable,'-m','pytest','-q','Libs/population_dynamics/test_pfc_history_tail_bound.py','-p','no:cacheprovider','--junitxml',str(output/'tests.xml')],'tests')
        tree = ET.parse(output/'tests.xml')
        suites = list(tree.getroot().iter('testsuite'))
        report['tests'] = {key:sum(int(node.get(key,0)) for node in suites) for key in ('tests','failures','errors','skipped')}
        source = repo/base/'profile_history/complete/run_605095_9'
        raw = (source/'delivery.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == '05250dc0bdd3e0d6c82c972a4d017e62e352adccea89d2c7849418f39b0863c6'
        delivery = json.loads(raw)
        assert delivery['status']=='COMPLETE_PFC_PROFILE_HISTORY_COMPLETE' and delivery['verified'] is True
        entries = {entry['path']:entry for entry in delivery['files']}
        report['source_delivery_sha256'] = hashlib.sha256(raw).hexdigest()
        key = 'B/half_quarter/field_4_cue'
        for name in ('report.json','selected_frame.json','classified_frame.json','integrated_frame.json','integration_details.tar'):
            relative = 'analysis/frames/'+key+'/'+name
            bound_copy(source/relative,work/'original/profile'/name,entries[relative]['sha256'])
        foundation = repo/base/'history_profile_census/prepare/run_604352'
        raw = (foundation/'delivery.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == 'b717ea0657c43d8bfc0376e4271961e34401b889a3abfbade7a0700ab90c0c9f'
        source_entries = {entry['path']:entry for entry in json.loads(raw)['files']}
        name = 'source_metadata/95002/quarter_B/config.yaml'
        bound_copy(foundation/name,work/'original/source_config.yaml',source_entries[name]['sha256'])
        common = [sys.executable,'-m','Libs.population_dynamics.pfc_history_tail_bound',
            '--config',str(work/'Libs/configs/neural_population_pfc_history_tail_bound.yaml'),
            '--source-config',str(work/'original/source_config.yaml'),
            '--profile',str(work/'original/profile')]
        command(common+['--output',str(output/'analysis')],'real_bound')
        result=json.loads((output/'analysis/report.json').read_text())
        assert result['source_rng_seed']==95002 and result['source_label']=='quarter_B'
        assert result['selected_physical_windows']==640
        assert all(0 <= row['upper_bound'] <= 1 for row in result['event_bounds'].values())
        assert not result['third_spike_probability_or_bound_qualified'] and not result['family_complete']
        previous = repo/base/'history_tail_bound/run_605431/analysis/report.json'
        previous_raw = previous.read_bytes()
        assert hashlib.sha256(previous_raw).hexdigest() == '707a0a16823ff18af3513d0a5c9180d14978f5a9eeb29a34dac0dfcecb2b2360'
        earlier = json.loads(previous_raw)
        assert result['source_rng_seed']==earlier['source_rng_seed'] and result['profile']==earlier['profile']
        assert result['window_sampling_bound']=='stratified_chernoff'
        report['method_comparison'] = {
            event: dict(historical_upper=earlier['event_bounds'][event]['upper_bound'],
                        aggregate_chernoff_upper=result['event_bounds'][event]['upper_bound'])
            for event in result['event_bounds']}
        report['historical_report_sha256']=hashlib.sha256(previous_raw).hexdigest()
        report['real_result']=result
        report['status']='COMPLETE_FINITE_ARCHIVED_HISTORY_TAIL_BOUND_VALIDATION'
        exit_code=0
    except Exception as exc:
        report.update(status='FAILED_FINITE_ARCHIVED_HISTORY_TAIL_BOUND_VALIDATION',error=dict(type=type(exc).__name__,message=str(exc)))
    finally:
        # Retain exactly staged code and the four small authenticated real inputs.
        if (work/'Libs').exists(): shutil.copytree(work/'Libs',output/'Libs')
        if (work/'original/source_config.yaml').exists(): shutil.copy2(work/'original/source_config.yaml',output/'source_config.yaml')
        report.update(exit_code=exit_code,wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('status','job_id','exit_code','wall_seconds')}))
raise SystemExit(exit_code)
PY_VALIDATION
/opt/conda/bin/python3 "$bound_launch_tmp/validate.py"

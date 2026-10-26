"""Prepare the pinned native PFC environment and validate clock admission gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request


MODULES = (
    'pfc_kernel_parameters','pfc_stream_validation','test_pfc_stream_validation','pfc_source_execution',
    'test_pfc_source_execution','pfc_scale_analysis','test_pfc_source_scale_analysis',
    'pfc_native_drive','pfc_deterministic_replay','pfc_input_pairing','pfc_delay_analysis',
    'pfc_response_analysis','pfc_reference_replay','pfc_source_comparison',
    'circuit_data','circuit_campaign','config','provenance','resource_probe',
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-parent-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    if os.environ.get('SLURMD_NODENAME')!='cpu05' or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('AUTHORIZED_CPU05_CPU_ONLY_ALLOCATION_REQUIRED')
    if args.output.exists():raise FileExistsError(args.output)
    repo=Path(__file__).resolve().parents[2];job=os.environ['SLURM_JOB_ID']
    work=Path(tempfile.mkdtemp(prefix='neuronsim-pfc-clock-preparation-'+job+'.',dir='/tmp'))
    prior=None;prior_files={};test_prior=None;test_archive=None
    if args.resume_from is not None:
        prior=json.loads((args.resume_from/'delivery.json').read_text())
        prior_files={row['path']:row for row in prior['files']}
        if prior['status']!='FAILED' or prior.get('error') not in ('install_brian failed:2','source_import failed:2'):
            raise ValueError('MATCHING_INTERRUPTED_INSTALL_REQUIRED')
        environment=Path(prior['native_environment'])
        test_archive=(args.resume_from if prior.get('tests_source_job')==prior['job_id']
                      else Path(prior.get('reused_preparation_archive',args.resume_from)))
        test_prior=json.loads((test_archive/'delivery.json').read_text())
        if environment!=Path(test_prior['native_environment']) or environment.parent!=Path('/tmp') or not environment.name.startswith('neuronsim-pfc-native-clock-env-') or not (environment/'bin/python').is_file():
            raise ValueError('ORIGINAL_NODE_LOCAL_ENVIRONMENT_REQUIRED')
    else:
        environment=Path('/tmp')/('neuronsim-pfc-native-clock-env-'+job)
    test_files={} if test_prior is None else {row['path']:row for row in test_prior['files']}
    record=dict(status='PREPARING',job_id=job,started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=str(work),native_environment=str(environment),commands=[],
        new_source_simulations=0,production_episodes=0,scientific_fidelity_qualified=False,
        full_clock_convergence_qualified=False)
    code=1
    try:
        package=work/'Libs/population_dynamics';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        for name in MODULES:shutil.copy2(repo/'Libs/population_dynamics'/(name+'.py'),package/(name+'.py'))
        (work/'phases').mkdir();(work/'scratch').mkdir()
        guard=repo/'Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py'
        shutil.copy2(guard,work/'run_guarded.py')
        shutil.copy2(Path(__file__),work/'launcher.py')
        env=dict(os.environ,PYTHONPATH=str(work/'Libs'),TMPDIR=str(work/'scratch'),
            XDG_CACHE_HOME=str(work/'cache'),MPLCONFIGDIR=str(work/'matplotlib'),
            PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PIP_CONFIG_FILE=os.devnull,PIP_INDEX_URL='https://pypi.org/simple',PIP_EXTRA_INDEX_URL='')
        def run(label,command):
            row=dict(label=label,command=[str(x) for x in command]);record['commands'].append(row)
            command_env=dict(env)
            if label=='install_brian':
                # Management-local proxy listeners are unavailable on the compute node.
                for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                    command_env.pop(key,None)
            result=subprocess.run([sys.executable,'-B',str(work/'run_guarded.py'),label,*row['command']],
                                  cwd=work,env=command_env)
            row['exit_code']=result.returncode
            if result.returncode:raise RuntimeError(label+' failed:'+str(result.returncode))
        tests=work/'run_tests.py'
        tests.write_text("""import json,sys,unittest
from pathlib import Path
suite=unittest.defaultTestLoader.loadTestsFromNames([
 'population_dynamics.test_pfc_source_execution',
 'population_dynamics.test_pfc_stream_validation',
 'population_dynamics.test_pfc_source_scale_analysis'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
Path('tests.json').write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),success=result.wasSuccessful())))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
""")
        if prior is None:
            run('tests',[sys.executable,'-B',tests])
            record['tests']=json.loads((work/'tests.json').read_text())
        else:
            changed_modules=[name for name in MODULES
                if digest(package/(name+'.py'))!=test_files['Libs/population_dynamics/'+name+'.py']['sha256']]
            for name in ('tests.json','clock_scope.json','phases/tests/receipt.json','phases/legacy_scope/receipt.json'):
                relative=name if name in test_files else 'reused_evidence/'+name
                path=test_archive/relative
                if digest(path)!=test_files[relative]['sha256']:raise ValueError('PRIOR_TEST_EVIDENCE_CHANGED')
                target=work/'reused_evidence'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            if not test_prior['tests']['success']:raise ValueError('PRIOR_TESTS_DID_NOT_PASS')
            record.update(tests=test_prior['tests'],tests_status='REUSED_PASS',tests_source_job=test_prior['job_id'],
                          reused_preparation_archive=str(test_archive),previous_attempt=str(args.resume_from))
            if changed_modules:
                run('tests',[sys.executable,'-B',tests])
                record.update(tests=json.loads((work/'tests.json').read_text()),tests_status='PASS_CURRENT_CHANGED_DEPENDENCIES',
                              tests_source_job=job,changed_dependencies=changed_modules)
        root=repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction'
        inputs={
            'legacy_validation/capture/native/report.json':root/'native_input_observation/native_validation_579585/capture/native/report.json',
            'input_replay.json':root/'native_input_observation/stream_numpy_584164/comparison/input_replay/comparison.json',
            'global_reference.json':root/'execution_profile/plain/report.json',
            'qualification.json':root/'native_input_observation/stream_numpy_584164/comparison/qualification.json',
        }
        for name,path in inputs.items():
            target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        scope=work/'reuse_scope.py'
        scope.write_text("""import json
from pathlib import Path
from population_dynamics.pfc_stream_validation import native_clock_scope
from population_dynamics.provenance import sha256_file
q=json.loads(Path('qualification.json').read_text())
assert q['global_reference_report_sha256']==sha256_file(Path('global_reference.json'))
assert q['input_replay_sha256']==sha256_file(Path('input_replay.json'))
result=native_clock_scope(Path('legacy_validation'),json.loads(Path('input_replay.json').read_text()),
 json.loads(Path('global_reference.json').read_text())['resolved_config'])
assert result==[.05]
Path('clock_scope.json').write_text(json.dumps(dict(status='PASS_REUSED_NATIVE_CLOCK_BINDING',validated_dt_ms=result)))
""")
        if prior is None:
            run('legacy_scope',[sys.executable,'-B',scope])
            record['legacy_clock_scope']=json.loads((work/'clock_scope.json').read_text())
            if environment.exists():raise FileExistsError(environment)
            run('create_environment',[args.native_parent_python,'-m','venv','--system-site-packages',environment])
        else:
            for name in inputs:
                if digest(work/name)!=test_files[name]['sha256']:raise ValueError('PRIOR_CLOCK_INPUT_CHANGED')
            record['legacy_clock_scope']=test_prior['legacy_clock_scope']
            record['environment_creation_reused']=True
        native=environment/'bin/python'
        smoke=root/'clock_refinement/environment_585089/clock_smoke.py'
        shutil.copy2(smoke,work/'clock_smoke.py')
        if prior is not None and prior['error']=='source_import failed:2':
            environment_archive=Path(prior.get('environment_install_and_smoke_reused_from',args.resume_from))
            installation=json.loads((environment_archive/'delivery.json').read_text())
            installation_files={row['path']:row for row in installation['files']}
            for name in ('pip_report.json','clock_smoke.py','phases/install_brian/receipt.json',
                         'phases/native_clock_smoke/receipt.json','phases/native_clock_smoke/command.log'):
                path=environment_archive/name
                if digest(path)!=installation_files[name]['sha256']:raise ValueError('PRIOR_ENVIRONMENT_EVIDENCE_CHANGED')
                target=work/'reused_environment'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            if digest(smoke)!=installation_files['clock_smoke.py']['sha256']:raise ValueError('CLOCK_SMOKE_CHANGED')
            passed={row['label']:row['exit_code'] for row in installation['commands']}
            if passed.get('install_brian')!=0 or passed.get('native_clock_smoke')!=0:raise ValueError('ENVIRONMENT_PHASE_NOT_COMPLETE')
            record['environment_install_and_smoke_reused_from']=str(environment_archive)
        else:
            run('install_brian',[native,'-m','pip','install','--no-deps','--retries','0','--timeout','20',
                                 '--report',work/'pip_report.json','brian2==2.5.4'])
            run('native_clock_smoke',[native,'-B',work/'clock_smoke.py'])
        source_root=root/'reference_size/source'
        manifest=json.loads((source_root/'source.json').read_text())
        entries={row['path']:row for row in manifest['files']}
        if manifest['revision']!='d7c4f07fa83a052829e40790fd1594955fe1c067':raise ValueError('PINNED_SOURCE_REVISION_REQUIRED')
        needed=('neurodynex3/__init__.py','neurodynex3/tools/__init__.py',
                'neurodynex3/tools/plot_tools.py','neurodynex3/tools/spike_tools.py',
                'neurodynex3/working_memory_network/wm_model.py')
        acquisition=[]
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for name in needed:
            target=work/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            available=source_root/name
            if args.resume_from is not None and (args.resume_from/'source'/name).is_file():
                available=args.resume_from/'source'/name
            if available.is_file():
                shutil.copy2(available,target);origin=str(available)
            else:
                origin='https://raw.githubusercontent.com/EPFL-LCN/neuronaldynamics-exercises/'+manifest['revision']+'/'+name
                with opener.open(origin,timeout=20) as response:
                    payload=response.read(entries[name]['bytes']+1)
                target.write_bytes(payload)
            if target.stat().st_size!=entries[name]['bytes'] or digest(target)!=entries[name]['sha256']:
                raise ValueError('PINNED_SOURCE_HELPER_MISMATCH:'+name)
            acquisition.append(dict(path=name,origin=origin,bytes=target.stat().st_size,sha256=digest(target)))
        shutil.copy2(source_root/'source.json',work/'source/source.json')
        (work/'source_acquisition.json').write_text(json.dumps(acquisition,indent=2))
        env['PYTHONPATH']=str(work/'Libs')+os.pathsep+str(work/'source')
        source_import=work/'source_import.py'
        source_import.write_text("""import hashlib,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path("source").resolve()))
path=Path('source/neurodynex3/working_memory_network/wm_model.py')
sha=hashlib.sha256(path.read_bytes()).hexdigest()
assert sha=='416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720'
spec=importlib.util.spec_from_file_location('pinned_pfc_source_import',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert callable(module.simulate_wm)
Path('source_import.json').write_text(json.dumps(dict(status='PASS_PINNED_SOURCE_IMPORT',source_entry_sha256=sha,source_simulations=0)))
""")
        run('source_import',[native,'-B',source_import])
        inventory=work/'inventory.py'
        inventory.write_text("""import json,sys
from importlib.metadata import version
from pathlib import Path
result=dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,
 brian2=version('brian2'),numpy=version('numpy'),scipy=version('scipy'),sympy=version('sympy'))
assert result['brian2']=='2.5.4' and result['numpy']=='1.26.4'
Path('native_environment.json').write_text(json.dumps(result,indent=2))
""")
        run('inventory',[native,'-B',inventory])
        run('parent_unchanged',[args.native_parent_python,'-c',
            "from importlib.metadata import version; assert version('brian2')=='2.9.0'; print('parent Brian2 unchanged:2.9.0')"])
        record['native_environment_details']=json.loads((work/'native_environment.json').read_text())
        record.update(status='PINNED_NATIVE_ENVIRONMENT_AND_CLOCK_GATES_READY',native_python=str(native),native_source=str(args.output/'source'))
        code=0
    except Exception as error:
        record.update(status='FAILED',error_type=type(error).__name__,error=str(error))
    record.update(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat())
    (work/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    args.output.mkdir(parents=True,exist_ok=False);files=[]
    for path in sorted(work.rglob('*')):
        relative=path.relative_to(work)
        if not path.is_file() or set(relative.parts)&{'scratch','cache','matplotlib','__pycache__'}:continue
        target=args.output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        sha=digest(path)
        if digest(target)!=sha:raise ValueError('CLOCK_PREPARATION_ARCHIVE_MISMATCH')
        files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=sha))
    (args.output/'delivery.json').write_text(json.dumps(dict(record,files=files),indent=2)+'\n')
    shutil.rmtree(work)
    print(json.dumps({key:record.get(key) for key in ('status','job_id','exit_code','native_python','error')}))
    return code


if __name__=='__main__':
    raise SystemExit(main())

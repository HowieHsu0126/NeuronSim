"""Prepare frozen fine-clock launcher inputs and perform import-only preflight.

No scientific launcher, test suite, simulator, or random-stream generator is run.
Historical test reuse is checked with each launcher's own run_tests function in a
matching local staging view whose execution callback always raises.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import yaml

SOURCE_FILES=('source.json','neurodynex3/__init__.py','neurodynex3/tools/__init__.py','neurodynex3/tools/plot_tools.py','neurodynex3/tools/spike_tools.py','neurodynex3/working_memory_network/wm_model.py')
MODULES={
 'native':('pfc_kernel_parameters','pfc_input_observation','pfc_input_capture','pfc_reference','pfc_reference_replay','pfc_factorized_source','pfc_initial_rng','circuit_data','circuit_campaign','config','provenance','resource_probe','test_pfc_input_observation','test_pfc_input_capture','test_pfc_reference','test_pfc_reference_resume','test_pfc_factorized_source'),
 'pairing':('pfc_input_pairing','test_pfc_input_pairing','provenance'),
 'stream':('pfc_kernel_parameters','pfc_deterministic_replay','test_pfc_deterministic_replay','pfc_native_drive','test_pfc_native_drive','pfc_source_execution','test_pfc_source_execution','pfc_stream_validation','test_pfc_stream_validation','pfc_input_pairing','provenance')}
LAUNCHERS={'native':'pfc_native_input_validation.sbatch','pairing':'pfc_input_pairing.sbatch','stream':'pfc_stream_validation.sbatch'}


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):result.update(data)
    return result.hexdigest()


def manifest(archive):
    archive=Path(archive);data=json.loads((archive/'delivery.json').read_text());result={}
    for row in data['files']:
        name=row['path'];require(name not in result and not Path(name).is_absolute()and '..'not in Path(name).parts,'INVALID_ARCHIVE_MANIFEST')
        result[name]=row
    return result


def checked(archive,entries,name):
    path=Path(archive)/name;row=entries[name]
    require(path.stat().st_size==row['bytes']and digest(path)==row['sha256'],'ARCHIVE_INPUT_CHANGED:'+name)
    return path


def validate_configs(snapshot,descriptors):
    records=[];by_clock={}
    for item in descriptors:
        config_path=Path(snapshot)/item['path'];require(digest(config_path)==item['sha256'],'FROZEN_CONFIG_CHANGED')
        config=yaml.safe_load(config_path.read_text());dt=item['dt_ms'];mode=item['mode']
        require(dt in (.025,.0125)and mode in ('independent','global'),'FIXED_FINE_CLOCKS_REQUIRED')
        fixed=dict(scope='source_reproduction',expected_source_revision='d7c4f07fa83a052829e40790fd1594955fe1c067',excitatory_neurons=1024,inhibitory_neurons=256,
                   weight_scaling_factor=2.,duration_ms=800.,readout_bin_ms=5.,sigma_weight_profile=20.,Jpos_excit2excit=2.3,synapse_execution='factorized_uniform_projections')
        require(config['dt_ms']==dt and all(config.get(k)==v for k,v in fixed.items()),'COMPLETE_FINE_CLOCK_PHYSICAL_TUPLE_REQUIRED')
        normalized=copy.deepcopy(config);normalized['dt_ms']=.05;normalized.pop('purpose',None)
        old=yaml.safe_load((Path(snapshot)/'Libs/configs'/('independent_streams.yaml'if mode=='independent'else'source_global.yaml')).read_text());old.pop('purpose',None)
        if mode=='independent':
            expected={(i,e,c)for i in (95001,95002)for e in (98401,98402)for c in ('omission','cue')}
            found=set()
            for row in config['cases']:
                condition='cue'if row['stimulus_strength_nA']==.06 else'omission'
                require(row['stimulus_strength_nA']in(0.,.06)and row['distractor_strength_nA']==0.,'FROZEN_INDEPENDENT_CONTROLS_REQUIRED')
                key=row['initial_rng_seed'],row['external_noise_seed'],condition
                require(row['case_id']==f'pfc_initial_{key[0]}_external_{key[1]}_{condition}','NEW_SEED_CASE_ID_MISMATCH');found.add(key)
            require(len(config['cases'])==8 and found==expected and config['random_stream_policy']=='independent_initial_external','EXACT_NEW_INDEPENDENT_SEED_ENVELOPE_REQUIRED')
            for row in normalized['cases']:
                row['external_noise_seed']={98401:96001,98402:96002}[row['external_noise_seed']]
                row['case_id']=row['case_id'].replace('98401','96001').replace('98402','96002')
            seeds=[95001,95002,98401,98402]
        else:
            require(config['source_rng_seed']==91001 and config['cases']==[dict(case_id='pfc_cue_only',distractor_strength_nA=0.),dict(case_id='pfc_cue_distractor',distractor_strength_nA=.03)],'EXACT_GLOBAL_CONTROL_ENVELOPE_REQUIRED')
            seeds=[91001]
        require(normalized==old,'UNDECLARED_CHANGE_FROM_HISTORICAL_SOURCE_CONFIG')
        require(not set(seeds)&set(range(96001,96129)),'PROTECTED_SEED_IN_PROSPECTIVE_CONFIG')
        require(round(800/dt)*dt==800 and round(5/dt)*dt==5,'SOURCE_CLOCK_ALIGNMENT_REQUIRED')
        by_clock.setdefault(dt,{})[mode]=item['path'];records.append(dict(**item,units=1280,duration_ms=800.,source_ticks=round(800/dt),seeds=seeds,validated=True))
    require(set(by_clock)=={.025,.0125}and all(set(v)=={'independent','global'}for v in by_clock.values()),'BOTH_COMPLETE_FINE_CLOCK_PAIRS_REQUIRED')
    return records,by_clock


def test_reuse(snapshot,view,kind,archive,entries):
    view=Path(view);package=view/'Libs/population_dynamics';package.mkdir(parents=True)
    for module in MODULES[kind]:shutil.copy2(Path(snapshot)/'Libs/population_dynamics'/(module+'.py'),package/(module+'.py'))
    fixture_proof=[]
    if kind=='native':
        for name in ('neural_population_pfc_independent_streams.yaml','neural_population_pfc_100k_source.yaml'):
            original=checked(archive,entries,'Libs/configs/'+name);staged=Path(snapshot)/'Libs/configs'/name
            require(digest(staged)==digest(original),'ORIGINAL_TEST_FIXTURE_CHANGED')
            fixture_proof.append(dict(path='Libs/configs/'+name,sha256=digest(staged)))
    launcher=(Path(snapshot)/'Output/neural_population_dynamics/operations'/LAUNCHERS[kind]).read_text()
    tree=ast.parse(launcher.split("<<'PYCODE'\n",1)[1].rsplit('PYCODE',1)[0]);function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='run_tests')
    test_code=next((node.value.value for node in ast.walk(tree)if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='test_code'for t in node.targets)and isinstance(node.value,ast.Constant)),None)
    previous=json.loads(checked(archive,entries,'execution.json').read_text());label='invariants'if kind=='stream'else'tests'
    rows=[row for row in previous['commands']if row['label']==label and row['status']=='PASS'and row.get('exit_code')==0];require(len(rows)==1,'EXACT_PRIOR_PASS_COMMAND_REQUIRED');command=rows[0]['command']
    def value(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.List):return [value(n)for n in node.elts]
        if isinstance(node,ast.Name)and node.id=='test_code':return test_code
        if isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='sys'and node.attr=='executable':return command[0]
        raise ValueError('UNSUPPORTED_TEST_COMMAND_EXPRESSION')
    calls=[node for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='run_tests']
    require(sum(value(call.args[0])==command and value(call.args[1])==label for call in calls)==1,'LAUNCHER_TEST_COMMAND_NO_LONGER_MATCHES')
    record={'commands':[]};report={'commands':[]}
    def forbidden(*args,**kwargs):raise RuntimeError('TEST_OR_SOURCE_EXECUTION_FORBIDDEN_DURING_PREPARATION')
    namespace=dict(Path=Path,json=json,shutil=shutil,sha256_file=digest,os=SimpleNamespace(environ={'NEURONSIM_PFC_TEST_ARCHIVE':str(archive)}),root=view,work=view,record=record,report=report,save=lambda:None,run=forbidden)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(archive/'launcher.sbatch'),'exec'),namespace)
    namespace['run_tests'](command,label);receipt=(record['commands']or report['commands'])[-1]
    require(receipt['status']=='REUSED_PASS'and receipt['command_started']is False,'REUSE_PROOF_MUST_NOT_EXECUTE_TESTS')
    checked(archive,entries,label+'.log')
    return dict(kind=kind,status='EXACT_UNCHANGED_TEST_REUSE_VERIFIED',command=command,command_started=False,module_count=len(MODULES[kind]),fixture_configs=fixture_proof,
                source_archive=str(archive),source_delivery_sha256=digest(archive/'delivery.json'),execution_sha256=entries['execution.json']['sha256'],log_sha256=entries[label+'.log']['sha256'])


def proposed_commands(snapshot,base,state,by_clock,archives,proofs):
    commands=[];shell=['#!/usr/bin/env bash','set -euo pipefail','# Prepared only; this file has NOT been executed.','# Confirm 597041 terminal success and native import receipt before manual launch.','previous_job=597041']
    test_python={row['kind']:row['command'][0]for row in proofs}
    for dt,tag in [(.025,'0p025'),(.0125,'0p0125')]:
        output=str(base/'executions'/('dt_'+tag+'ms'))
        native_var='native_'+tag;pair_var='pairing_'+tag;stream_var='stream_'+tag
        for kind,var,dependency in [('native',native_var,'${previous_job}'),('pairing',pair_var,'${'+native_var+'}'),('stream',stream_var,'${'+pair_var+'}')]:
            env=dict(NEURONSIM_PROJECT_ROOT=str(snapshot),NEURONSIM_PFC_PYTHON=test_python[kind],NEURONSIM_PFC_SCRATCH='/tmp',NEURONSIM_PFC_OUTPUT_BASE=output,NEURONSIM_PFC_TEST_ARCHIVE=str(archives[kind]),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
            if kind=='native':env.update(NEURONSIM_PFC_SOURCE=str(snapshot/'source'),NEURONSIM_PFC_CONFIG=str(snapshot/by_clock[dt]['independent']),NEURONSIM_PFC_GLOBAL_CONFIG=str(snapshot/by_clock[dt]['global']),NEURONSIM_PFC_PLAIN_REFERENCE='execute')
            else:env['NEURONSIM_PFC_VALIDATION']=output+'/native_validation_${'+native_var+'}'
            if kind=='stream':env.update(NEURONSIM_PFC_PAIRING=output+'/pairing_${'+pair_var+'}',NEURONSIM_PFC_GLOBAL_REFERENCE=output+'/native_validation_${'+native_var+'}/global_reference',NEURONSIM_PFC_REPLAY_BACKEND='numpy')
            args=['sbatch','--parsable','--job-name=NrnSim_GainClock'+tag+kind.capitalize(),'--partition='+('gpu'if kind=='native'else'cpu'),'--nodelist='+('cpu03'if kind=='native'else'cpu04'),'--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G','--gres=none','--no-requeue','--kill-on-invalid-dep=yes','--dependency=afterok:'+dependency,
                  '--output=/tmp/neuronsim-gain-clock-'+tag+'-'+kind+'-%j.out','--error=/tmp/neuronsim-gain-clock-'+tag+'-'+kind+'-%j.err','--export=ALL,'+','.join(k+'='+v for k,v in env.items()),str(snapshot/'Output/neural_population_dynamics/operations'/LAUNCHERS[kind])]
            # Only predeclared shell variables are expanded; every static argument is quoted.
            quoted=[]
            for arg in args:
                if '${'in arg:quoted.append('"'+arg.replace('\\','\\\\').replace('"','\\"').replace('`','\\`')+'"')
                else:quoted.append(shlex.quote(arg))
            shell.append(var+'=$('+ ' '.join(quoted)+')');commands.append(dict(clock_ms=dt,kind=kind,argv=args,source_executions=26 if kind=='native'else 10 if kind=='stream'else 0,submitted=False))
        shell.append('previous_job=${'+stream_var+'}')
    return commands,'\n'.join(shell)+'\n'


PROBE=r"""import hashlib,importlib,json,os,platform,random,sys
from pathlib import Path
import numpy as np
import brian2,scipy
snapshot=Path(sys.argv[1]);expected=snapshot/'source/neurodynex3/working_memory_network/wm_model.py'
simulation_calls=[]
def profile(frame,event,arg):
    if event=='call'and Path(frame.f_code.co_filename)==expected and frame.f_code.co_name=='simulate_wm':
        simulation_calls.append('simulate_wm');raise RuntimeError('SOURCE_SIMULATION_FORBIDDEN_IN_IMPORT_PREFLIGHT')
before=np.random.get_state();python_before=random.getstate();sys.setprofile(profile)
try:
    wm=importlib.import_module('neurodynex3.working_memory_network.wm_model')
    for name in ('pfc_reference','pfc_input_capture','pfc_input_observation'):importlib.import_module('population_dynamics.'+name)
finally:sys.setprofile(None)
after=np.random.get_state()
unchanged=before[0]==after[0]and np.array_equal(before[1],after[1])and before[2:]==after[2:]
assert Path(wm.__file__).resolve()==expected.resolve()and callable(wm.simulate_wm)
assert unchanged and python_before==random.getstate()and not simulation_calls
report=dict(status='PASS_NATIVE_SOURCE_IMPORT_ONLY',interpreter=sys.executable,python=platform.python_version(),brian2=brian2.__version__,numpy=np.__version__,scipy=scipy.__version__,
 source_module=str(wm.__file__),source_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),simulation_calls=0,source_executions=0,
 numpy_global_rng_unchanged=True,python_global_rng_unchanged=True,unit_tests_executed=False,protected_seed_streams_generated=False)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
"""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--mode',choices=['prepare','import'],required=True);parser.add_argument('--preparation',type=Path)
    args=parser.parse_args();repo=args.project_root;job=os.environ['SLURM_JOB_ID'];node=socket.gethostname().split('.')[0]
    require(node==('cpu04'if args.mode=='prepare'else'cpu03'),'AUTHORIZED_PREPARATION_NODE_REQUIRED')
    base=repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/kernel_parameterization/recurrent_gain/fine_clock_preparation'
    archive=base/(('run_'if args.mode=='prepare'else'import_')+job);require(not archive.exists(),'IMMUTABLE_OUTPUT_ALREADY_EXISTS')
    work=Path(tempfile.mkdtemp(prefix='neuronsim-fine-clock-'+job+'.',dir='/tmp'));snapshot=work/'snapshot';snapshot.mkdir();inputs=[];status='FAILED';code=1;started=time.monotonic()
    def stage(origin,relative,expected=None):
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target);sha=digest(target)
        require(expected is None or sha==expected,'STAGED_CONTENT_CHANGED:'+str(relative));inputs.append(dict(source=str(origin),path=str(relative),bytes=target.stat().st_size,sha256=sha));return target
    try:
        stage(Path(__file__),'preparation.py');stage(Path(os.environ['NEURONSIM_FINE_CLOCK_SUBMITTED_BATCH']),'preparation.sbatch')
        if args.mode=='prepare':
            state=json.loads((repo/'Output/neural_population_dynamics/operations/continuation_resources_581147.json').read_text())['pfc_clock_refinement']['recurrent_gain_fine_clock_preparation']
            archives={kind:repo/state[kind+'_tests_reuse_archive']for kind in MODULES};all_entries={};duplicates=[]
            for kind,origin in archives.items():
                entries=manifest(origin);all_entries[kind]=entries
                for module in MODULES[kind]:
                    name='Libs/population_dynamics/'+module+'.py';original=checked(origin,entries,name);target=snapshot/name
                    if target.exists():require(digest(target)==entries[name]['sha256'],'DUPLICATE_MODULE_BYTES_DIFFER');duplicates.append(dict(module=name,archive=str(origin),sha256=entries[name]['sha256']))
                    else:stage(original,Path('snapshot')/name,entries[name]['sha256'])
                name='Output/neural_population_dynamics/operations/'+LAUNCHERS[kind]
                stage(checked(origin,entries,'launcher.sbatch'),Path('snapshot')/name,entries['launcher.sbatch']['sha256'])
            native=archives['native'];entries=all_entries['native']
            for name in ('independent_streams.yaml','source_global.yaml','neural_population_pfc_independent_streams.yaml','neural_population_pfc_100k_source.yaml'):
                relative='Libs/configs/'+name;stage(checked(native,entries,relative),Path('snapshot')/relative,entries[relative]['sha256'])
            source_meta=json.loads(checked(native,entries,'source/source.json').read_text());source_entries={r['path']:r for r in source_meta['files']}
            require(source_meta['revision']=='d7c4f07fa83a052829e40790fd1594955fe1c067','PINNED_SOURCE_REVISION_REQUIRED')
            for name in SOURCE_FILES:
                relative='source/'+name;original=checked(native,entries,relative)
                if name!='source.json':require(digest(original)==source_entries[name]['sha256'],'SOURCE_HELPER_MANIFEST_MISMATCH')
                stage(original,Path('snapshot')/relative,entries[relative]['sha256'])
            for item in state['configs']:stage(repo/item['path'],Path('snapshot')/item['path'],item['sha256'])
            records,clocks=validate_configs(snapshot,state['configs']);proofs=[]
            for kind in MODULES:proofs.append(test_reuse(snapshot,work/'reuse_views'/kind,kind,archives[kind],all_entries[kind]))
            future_snapshot=archive/'snapshot';commands,shell=proposed_commands(future_snapshot,base,state,clocks,archives,proofs)
            (work/'proposed_commands.json').write_text(json.dumps(commands,indent=2)+'\n');(work/'proposed_commands.sh').write_text(shell)
            subprocess.run(['bash','-n',str(work/'proposed_commands.sh')],check=True)
            for kind in MODULES:subprocess.run(['bash','-n',str(snapshot/'Output/neural_population_dynamics/operations'/LAUNCHERS[kind])],check=True)
            report=dict(status='PREPARED_NATIVE_FINE_CLOCK_SNAPSHOT',configs=records,test_reuse=proofs,duplicate_modules_verified=duplicates,
                source_revision=source_meta['revision'],source_helpers=list(SOURCE_FILES),native_python=state['native_python'],native_import_verified=False,
                future_source_executions_per_clock=dict(native=26,numpy=10),source_jobs_submitted=[],unit_tests_executed=False,
                prospective_external_noise_seeds=[98401,98402],initial_seeds=[95001,95002],global_seed=91001,
                source_executions=0,protected_seed_streams_generated=False,confirmation_execution_released=False,production_episodes=0)
            status=report['status'];code=0
        else:
            require(args.preparation is not None,'COMPLETED_PREPARATION_REQUIRED');entries=manifest(args.preparation)
            prepared=json.loads(checked(args.preparation,entries,'report.json').read_text());require(prepared['status']=='PREPARED_NATIVE_FINE_CLOCK_SNAPSHOT','SUCCESSFUL_SNAPSHOT_PREPARATION_REQUIRED')
            for name,row in entries.items():
                if name.startswith('snapshot/')and not name.endswith('.sbatch'):stage(checked(args.preparation,entries,name),name,row['sha256'])
            probe=work/'import_probe.py';probe.write_text(PROBE);output=work/'import_result.json'
            for name in ('tmp','mpl_cache','cache'):(work/name).mkdir()
            env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(snapshot/'Libs'),str(snapshot/'source')]),NEURONSIM_PFC_SOURCE=str(snapshot/'source'),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(work/'mpl_cache'),XDG_CACHE_HOME=str(work/'cache'),TMPDIR=str(work/'tmp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            command=[prepared['native_python'],'-B',str(probe),str(snapshot),str(output)]
            with(work/'import.stdout').open('w')as out,(work/'import.stderr').open('w')as err:result=subprocess.run(command,cwd=work,env=env,stdout=out,stderr=err)
            require(result.returncode==0,'NATIVE_SOURCE_IMPORT_PREFLIGHT_FAILED:'+str(result.returncode));report=json.loads(output.read_text())
            require(report['source_sha256']=='416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720','IMPORTED_PINNED_SOURCE_CHANGED')
            report.update(preparation=str(args.preparation),preparation_delivery_sha256=digest(args.preparation/'delivery.json'),command=command,node=node)
            status=report['status'];code=0
    except Exception as error:
        report=dict(status='FAILED_FINE_CLOCK_'+args.mode.upper(),error=str(error),traceback=traceback.format_exc(),source_executions=0,source_jobs_submitted=[],protected_seed_streams_generated=False)
    report.update(job_id=job,node=node,mode=args.mode,exit_code=code,inputs=inputs,wall_seconds=time.monotonic()-started)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');delivered=False
    try:
        archive.mkdir(parents=True,exist_ok=False);files=[]
        for path in work.rglob('*'):
            if not path.is_file()or set(path.relative_to(work).parts)&{'reuse_views','mpl_cache','cache','tmp','__pycache__'}:continue
            relative=path.relative_to(work);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sha=digest(path);require(digest(target)==sha,'DELIVERY_MISMATCH')
            files.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=sha))
        (archive/'delivery.json').write_text(json.dumps(dict(status=report['status'],exit_code=code,verified=True,files=files),indent=2)+'\n');delivered=True
        print(json.dumps(dict(status=report['status'],archive=str(archive),exit_code=code,files=len(files))))
    finally:
        if delivered:shutil.rmtree(work)
        else:print('COPYBACK_FAILED_SCRATCH_PRESERVED '+str(work),file=sys.stderr)
    return code

if __name__=='__main__':raise SystemExit(main())

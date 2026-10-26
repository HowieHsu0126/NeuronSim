#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_HistorySummary
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-history-summary-%j.out
#SBATCH --error=/tmp/neuronsim-history-summary-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_SUMMARY_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_SUMMARY'
from pathlib import Path
from datetime import datetime,timezone
import ast,csv,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time,traceback
import xml.etree.ElementTree as ET
import yaml
import numpy as np
repo=Path(os.environ['NEURONSIM_PROJECT_ROOT']);job=os.environ['SLURM_JOB_ID']
assert (repo/'PROJECT_PREFIX').read_text().strip()=='NrnSim'
mode=os.environ.get('NEURONSIM_SUMMARY_MODE','prepare');assert mode in ('prepare','complete')
task=os.environ.get('NEURONSIM_SUMMARY_TASK','history');assert task in ('history','clock_comparison','timing_prediction','timing_census','stratified_history','stratified_summary','design_comparison','profile_census','profile_summary','profile_history','profile_readback','profile_coverage')
expected_node='cpu03' if task in ('timing_census','stratified_history','profile_census','profile_history') and mode=='complete' else 'cpu04'
assert socket.gethostname().split('.')[0]==expected_node and not os.environ.get('SLURM_JOB_GPUS')
assert task not in ('timing_prediction','design_comparison','profile_summary') or mode=='prepare'
name={'history':'first_passage_history_summary','clock_comparison':'first_passage_clock_comparison','timing_prediction':'history_timing_prediction','timing_census':'history_timing_census','stratified_history':'stratified_history','stratified_summary':'stratified_history_summary','design_comparison':'history_design_comparison','profile_census':'history_profile_census','profile_summary':'history_profile_summary','profile_history':'profile_history','profile_readback':'profile_history_readback','profile_coverage':'profile_history_coverage'}[task]
status_name={'history':'PFC_HISTORY_SUMMARY','clock_comparison':'PFC_CLOCK_HISTORY_COMPARISON','timing_prediction':'PFC_HISTORY_TIMING_PREDICTION','timing_census':'PFC_HISTORY_TIMING_CENSUS','stratified_history':'PFC_STRATIFIED_HISTORY','stratified_summary':'PFC_STRATIFIED_HISTORY_SUMMARY','design_comparison':'PFC_HISTORY_DESIGN_COMPARISON','profile_census':'PFC_HISTORY_PROFILE_CENSUS','profile_summary':'PFC_HISTORY_PROFILE_SUMMARY','profile_history':'PFC_PROFILE_HISTORY','profile_readback':'PFC_PROFILE_HISTORY_READBACK','profile_coverage':'PFC_PROFILE_HISTORY_COVERAGE'}[task]
if task in ('stratified_history','profile_census','profile_history','profile_readback','profile_coverage') and 'SLURM_ARRAY_JOB_ID' in os.environ:
    job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
archive=repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement'/name/mode/('run_'+job)
assert not archive.exists()
work=Path(tempfile.mkdtemp(prefix='neuronsim-history-summary-'+job+'.',dir='/tmp'))
(work/'scratch').mkdir();env=dict(os.environ,PYTHONPATH=str(work),TMPDIR=str(work/'scratch'))
config=Path('Libs/configs/neural_population_pfc_'+name+'.yaml')
config_source=Path(os.environ.get('NEURONSIM_SUMMARY_CONFIG_SOURCE',str(config)))
if not config_source.is_absolute():config_source=repo/config_source
module=Path('Libs/population_dynamics/pfc_'+name+'.py')
test=Path('Libs/population_dynamics/test_pfc_'+name+'.py')
modules=[module];tests=[test]
if task=='clock_comparison':
    modules.append(Path('Libs/population_dynamics/pfc_first_passage_history_summary.py'))
    tests.append(Path('Libs/population_dynamics/test_pfc_first_passage_history_summary.py'))
if task=='stratified_summary':
    modules.append(Path('Libs/population_dynamics/pfc_stratified_history.py'))
if task=='stratified_history':
    modules.extend(Path('Libs/population_dynamics')/name for name in ('pfc_first_passage_history.py','pfc_first_passage_clock_comparison.py'))
    tests.extend(Path('Libs/population_dynamics')/name for name in ('test_pfc_first_passage_history.py','test_pfc_first_passage_history_suppression.py'))
if task=='profile_census':
    modules.append(Path('Libs/population_dynamics/pfc_history_timing_census.py'))
    tests.append(Path('Libs/population_dynamics/test_pfc_history_timing_census.py'))
if task=='profile_summary':
    modules.extend(Path('Libs/population_dynamics')/name for name in ('pfc_history_profile_census.py','pfc_history_timing_census.py'))
if task=='profile_history':
    modules.extend(Path('Libs/population_dynamics')/name for name in ('pfc_first_passage_history.py','pfc_stratified_history.py','pfc_first_passage_clock_comparison.py','pfc_first_passage_history_summary.py','pfc_history_profile_census.py','pfc_history_timing_census.py','test_pfc_history_timing_census.py'))
    tests.extend(Path('Libs/population_dynamics')/name for name in ('test_pfc_first_passage_history.py','test_pfc_first_passage_fine_history.py','test_pfc_stratified_history.py'))
if task=='profile_readback':
    modules.extend(Path('Libs/population_dynamics')/name for name in ('pfc_profile_history.py','test_pfc_profile_history.py'))
expected_tests=49 if task=='profile_coverage' else 35 if task=='profile_readback' else 55 if task=='profile_history' else 5 if task=='profile_summary' else 18 if task=='profile_census' else 25 if task=="stratified_history" else 5 if task=="design_comparison" else 14 if task=="timing_census" else 6 if task=="stratified_summary" else 4 if task=="timing_prediction" else 3*len(tests)
inputs=[];commands=[];status='FAILED_'+status_name+'_'+mode.upper();error=None;exit_code=1;delivered=False;begin=time.perf_counter()

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()
def copy(source,relative,expected=None):
    target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    sha=digest(target)
    if expected is not None and sha!=expected:raise ValueError('HISTORY_SUMMARY_INPUT_CHANGED:'+str(relative))
    inputs.append(dict(source=str(source),path=str(relative),sha256=sha))

def phase(name,args):
    start=time.perf_counter()
    with (work/(name+'.stdout')).open('w') as out,(work/(name+'.stderr')).open('w') as err:
        result=subprocess.run(args,cwd=work,env=env,stdout=out,stderr=err)
    commands.append(dict(phase=name,command=args,exit_code=result.returncode,wall_seconds=time.perf_counter()-start))
    print(json.dumps(commands[-1]),flush=True)
    if result.returncode:raise RuntimeError('HISTORY_SUMMARY_'+name.upper()+'_FAILED:'+str(result.returncode))

try:
    copy(config_source,config,os.environ['NEURONSIM_SUMMARY_CONFIG_SHA256'])
    settings=yaml.safe_load((work/config).read_text());assert settings['schema']==('pfc_history_profile_census_v1' if task=='profile_summary' else 'pfc_'+name+'_v1')
    for relative in (Path('Libs/__init__.py'),Path('Libs/population_dynamics/__init__.py'),Path('Libs/population_dynamics/resource_probe.py')):
        if (repo/relative).exists():copy(repo/relative,relative)
    if mode=='prepare':
        for relative in modules+tests:copy(repo/relative,relative)
    else:
        prepared=Path(os.environ['NEURONSIM_SUMMARY_PREPARATION'])
        copy(prepared/'delivery.json',Path('preparation_delivery.json'),os.environ['NEURONSIM_SUMMARY_PREPARATION_SHA256'])
        pd=json.loads((work/'preparation_delivery.json').read_text());assert pd['verified'] and pd['status']=='COMPLETE_'+status_name+'_PREPARE'
        pe={r['path']:r for r in pd['files']};assert digest(work/config)==pe[str(config)]['sha256']
        for relative in modules+tests:copy(prepared/relative,relative,pe[str(relative)]['sha256'])
        copy(prepared/'report.json',Path('preparation_report.json'),pe['report.json']['sha256'])
        prior=json.loads((work/'preparation_report.json').read_text())
        prior_test_count=sum(int(r['tests']) for r in prior['test_suites'])
        if (task=='profile_coverage' and prior_test_count in (45,49)) or (task=='timing_census' and prior_test_count in (6,14)) or (task=='stratified_history' and prior_test_count in (13,25)):
            # The archived six-test preparation and current shared-census tests are both supported.
            expected_tests=prior_test_count
        assert prior['exit_code']==0 and prior_test_count==expected_tests
        assert all(int(r['failures'])==int(r['errors'])==int(r['skipped'])==0 for r in prior['test_suites'])
        commands.append(dict(phase='variance_tests_reused',tests=expected_tests,archive=str(prepared),exit_code=0))
        if task=='profile_history' and 'acceleration_delivery.json' in pe:
            copy(prepared/'acceleration_delivery.json',Path('acceleration_delivery.json'),pe['acceleration_delivery.json']['sha256'])
            copy(prepared/'acceleration_report.json',Path('acceleration_report.json'),pe['acceleration_report.json']['sha256'])
            ar=json.loads((work/'acceleration_report.json').read_text())
            assert ar['exit_code']==0 and ar['prefix_state_and_rng_exact'] is True
            for name in ('pfc_first_passage_reference.py','pfc_first_passage_history.py','pfc_clock_coupling.py'):
                relative=Path('Libs/population_dynamics')/name
                copy(prepared/relative,relative,pe[str(relative)]['sha256'])
    if mode=='complete' and task=='profile_history' and 'passage_parity_report.json' in pe:
        copy(prepared/'passage_parity_report.json',Path('passage_parity_report.json'),pe['passage_parity_report.json']['sha256'])
        parity=json.loads((work/'passage_parity_report.json').read_text())
        assert parity['status']=='PASS_PFC_PASSAGE_STATE_AND_IMPORTANCE_PARITY' and parity['exit_code']==0
        assert len(parity['states'])==6 and all(row['exact'] for row in parity['states'])
        assert len(parity['integrations'])==2 and all(row['all_scores_histograms_witnesses_rng_exact'] for row in parity['integrations'])
        for name,key in (('pfc_first_passage_probability.py','canonical_probability_sha256'),
                         ('pfc_first_passage_importance.py','canonical_importance_sha256')):
            relative=Path('Libs/population_dynamics')/name
            assert pe[str(relative)]['sha256']==parity[key]
            copy(prepared/relative,relative,pe[str(relative)]['sha256'])
        commands.append(dict(phase='first_passage_parity_reused',job_id=parity['job_id'],exit_code=0))
    if mode=='complete' and task=='profile_history' and 'projection_parity_report.json' in pe:
        copy(prepared/'projection_parity_report.json',Path('projection_parity_report.json'),pe['projection_parity_report.json']['sha256'])
        projection=json.loads((work/'projection_parity_report.json').read_text())
        assert projection['status']=='PASS_PFC_PROJECTION_STATE_AND_RNG_PARITY' and projection['exit_code']==0
        assert len(projection['clocks'])==3 and all(row['captured_and_terminal_states_and_rng_exact'] for row in projection['clocks'])
        relative=Path('Libs/population_dynamics/pfc_first_passage_reference.py')
        assert pe[str(relative)]['sha256']==projection['canonical_reference_sha256']
        copy(prepared/relative,relative,pe[str(relative)]['sha256'])
        for name in ('pfc_deterministic_replay.py','pfc_kernel_parameters.py'):
            relative=Path('Libs/population_dynamics')/name
            copy(prepared/relative,relative,pe[str(relative)]['sha256'])
        commands.append(dict(phase='reference_projection_parity_reused',job_id=projection['job_id'],exit_code=0))
    if task in ('timing_census','stratified_history','stratified_summary','design_comparison','profile_census','profile_summary','profile_history','profile_readback','profile_coverage'):
        foundation=settings['source_preparation'] if task in ('profile_readback','profile_coverage') else settings['preparation'] if task=='stratified_summary' else settings['foundations'];origin=repo/foundation['archive']
        copy(origin/'delivery.json',Path('foundation_delivery.json'),foundation['delivery_sha256'])
        fd=json.loads((work/'foundation_delivery.json').read_text())
        assert fd['verified'] and fd['status']==('COMPLETE_PFC_PROFILE_HISTORY_READBACK_PREPARE' if task=='profile_coverage' else 'COMPLETE_PFC_PROFILE_HISTORY_PREPARE' if task=='profile_readback' else 'COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_PREPARE' if task=='design_comparison' else 'COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE' if task=='stratified_summary' else 'COMPLETE_PFC_FIRST_PASSAGE_HISTORY_SUPPRESSION_PREPARE')
        for entry in fd['files']:
            relative=Path(entry['path'])
            if relative.suffix=='.py' and relative.parts[0]=='Libs' and (not relative.name.startswith('test_') or task in ('stratified_summary','profile_history','profile_readback','profile_coverage') or (task=='stratified_history' and relative.name=='test_pfc_first_passage_suppression.py')) and not (work/relative).exists():
                copy(origin/relative,relative,entry['sha256'])
    acceleration_receipt=os.environ.get('NEURONSIM_SUMMARY_ACCELERATION_RECEIPT')
    if acceleration_receipt:
        assert task=='profile_history' and mode=='prepare'
        accelerated=Path(acceleration_receipt)
        copy(accelerated/'delivery.json',Path('acceleration_delivery.json'),os.environ['NEURONSIM_SUMMARY_ACCELERATION_SHA256'])
        ad=json.loads((work/'acceleration_delivery.json').read_text())
        assert ad['verified'] and ad['status']=='PASS_PFC_REFERENCE_PERFORMANCE_VALIDATION'
        ae={entry['path']:entry for entry in ad['files']}
        copy(accelerated/'report.json',Path('acceleration_report.json'),ae['report.json']['sha256'])
        ar=json.loads((work/'acceleration_report.json').read_text())
        assert ar['exit_code']==0 and ar['prefix_state_and_rng_exact'] is True
        assert all(suite['failures']==suite['errors']==suite['skipped']==0 for suite in ar['tests'])
        for name in ('pfc_first_passage_reference.py','pfc_first_passage_history.py','pfc_clock_coupling.py'):
            relative=Path('Libs/population_dynamics')/name
            snapshot='candidate/'+str(relative)
            copy(accelerated/snapshot,relative,ae[snapshot]['sha256'])
        commands.append(dict(phase='acceleration_receipt_bound',archive=str(accelerated),tests=sum(x['tests'] for x in ar['tests']),exit_code=0))
    if task=='stratified_history':
        census_source=settings['accepted_census'];census_origin=repo/census_source['archive']
        copy(census_origin/'delivery.json',Path('census_code_delivery.json'),census_source['delivery_sha256'])
        census_code=json.loads((work/'census_code_delivery.json').read_text())
        assert census_code['verified'] and census_code['status']=='COMPLETE_PFC_HISTORY_TIMING_CENSUS_COMPLETE'
        cc={row['path']:row for row in census_code['files']}
        relative=Path('Libs/population_dynamics/pfc_history_timing_census.py')
        copy(census_origin/relative,relative,cc[str(relative)]['sha256'])
    copy(Path(os.environ['NEURONSIM_SUMMARY_LAUNCHER']),Path('launcher.sbatch'))
    sys.path.insert(0,str(work))
    from Libs.population_dynamics.resource_probe import probe
    health=probe(work/'resource_preflight.json',4);assert health['status']=='PASS',health['reasons']
    if mode=='prepare':
        test_receipt=os.environ.get('NEURONSIM_SUMMARY_TEST_RECEIPT')
        if test_receipt:
            assert task in ('stratified_history','design_comparison')
            tested=Path(test_receipt)
            copy(tested/'delivery.json',Path('test_receipt_delivery.json'),os.environ['NEURONSIM_SUMMARY_TEST_RECEIPT_SHA256'])
            tr=json.loads((work/'test_receipt_delivery.json').read_text());assert tr['verified']
            tf={row['path']:row for row in tr['files']}
            copy(tested/'report.json',Path('test_receipt_report.json'),tf['report.json']['sha256'])
            passed=json.loads((work/'test_receipt_report.json').read_text())
            assert any(command['phase']=='tests' and command['exit_code']==0 for command in passed['commands'])
            assert sum(int(row['tests']) for row in passed['test_suites'])==expected_tests
            assert all(int(row['failures'])==int(row['errors'])==int(row['skipped'])==0 for row in passed['test_suites'])
            if task=='stratified_history':
                assert digest(work/config)==tf[str(config)]['sha256']
                tested_paths=list((work/'Libs').rglob('*.py'))
            else:
                assert passed['configuration_read_by_tests'] is False
                tested_paths=[work/module,work/test,work/'Libs/population_dynamics/pfc_first_passage_history_summary.py']
            for path in tested_paths:
                relative=str(path.relative_to(work));assert digest(path)==tf[relative]['sha256'], 'SUMMARY_TESTED_CODE_CHANGED:'+relative
            copy(tested/'tests.xml',Path('tests.xml'),tf['tests.xml']['sha256'])
            commands.append(dict(phase='tests_reused',archive=str(tested),tests=expected_tests,exit_code=0))
        else:
            phase('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--import-mode=importlib',*[str(t) for t in tests],
                           '--basetemp='+str(work/'test_tmp'),'--junitxml='+str(work/'tests.xml')])
    if task=='profile_coverage':
        from Libs.population_dynamics.pfc_profile_history_coverage import run
        result=run(settings,repo,work,copy,mode,fd)
        commands.append(dict(phase='profile_coverage_'+mode,status=result['status'],exit_code=0))
    elif task=='profile_readback':
        from Libs.population_dynamics.pfc_profile_history_readback import read_pair, readback_requests
        from Libs.population_dynamics.pfc_history_profile_census import write_json, BOUNDARY
        source_prepared=repo/settings['source_preparation']['archive']
        source_entries={row['path']:row for row in fd['files']}
        source_config='Libs/configs/neural_population_pfc_profile_history.yaml'
        assert source_entries[source_config]['sha256']==settings['source_config_sha256']
        copy(source_prepared/source_config,Path('source_config.yaml'),settings['source_config_sha256'])
        history_settings=yaml.safe_load((work/'source_config.yaml').read_text())
        requests=readback_requests(settings,history_settings)
        if mode=='prepare':
            write_json(work/'analysis/report.json',dict(status='PREPARED_PFC_PROFILE_HISTORY_READBACK',
                source_execution_verified=False,
                required_source_jobs=len({spec['archive'] for request in requests for spec in request['source_runs'].values()}),
                required_readback_groups=len(requests),
                readback_requests=[dict(source_rng_seed=request['source_rng_seed'],group_name=request.get('group_name'),
                    profile_keys=request['profile_keys']) for request in requests],**BOUNDARY))
        else:
            index=int(os.environ.get('SLURM_ARRAY_TASK_ID',os.environ.get('NEURONSIM_PROFILE_READBACK_INDEX','0')))
            assert 0<=index<len(requests), 'PROFILE_READBACK_INDEX'
            settings=requests[index]
            for name in ('source_index.json','census_run_index.json'):
                copy(source_prepared/name,Path(name),source_entries[name]['sha256'])
            source_index=json.loads((work/'source_index.json').read_text())
            census_index=json.loads((work/'census_run_index.json').read_text())
            seed=settings['source_rng_seed'];bindings={};source_costs={}
            for clock,spec in settings['source_runs'].items():
                origin=repo/spec['archive'];target=Path('source_inputs')/clock
                copy(origin/'delivery.json',target/'delivery.json')
                receipt=json.loads((work/target/'delivery.json').read_text())
                assert receipt['verified'] and receipt['status']=='COMPLETE_PFC_PROFILE_HISTORY_COMPLETE'
                entries={row['path']:row for row in receipt['files']}
                assert entries[source_config]['sha256']==settings['source_config_sha256']
                for name in ('report.json','analysis/report.json','runtime_config.json'):
                    copy(origin/name,target/name,entries[name]['sha256'])
                wrapper=json.loads((work/target/'report.json').read_text());assert wrapper['exit_code']==0 and wrapper['error'] is None
                source_report=json.loads((work/target/'analysis/report.json').read_text())
                assert source_report['status']=='COMPLETE_PFC_PROFILE_SOURCE_HISTORY' and source_report['source_rng_seed']==seed
                expected_profile_keys=spec.get('expected_profile_keys',settings['profile_keys'])
                assert source_report['source_label']==spec['source_label']
                assert source_report['requested_profile_keys']==source_report['completed_profile_keys']==expected_profile_keys
                assert source_report['pilot'] is (not settings['full_source_requests'])
                assert set(settings['profile_keys'])<=set(expected_profile_keys)
                binding=json.loads((work/target/'runtime_config.json').read_text())['source_binding']
                registered=source_index['source_blocks'][str(seed)]['sources'][spec['source_label']]
                assert all(binding[k]==registered[k] for k in ('archive','delivery_sha256','refinement_seeds'))
                bindings[clock]=binding;source_costs[clock]=dict(history_execution_archive=spec['archive'],history_execution_job_id=receipt['job_id'],reconstruction_seconds=source_report['reconstruction_seconds'],wall_seconds=source_report['wall_seconds'],peak_rss_kib=source_report['peak_rss_kib'],source_reconstruction_calls=source_report['source_reconstruction_calls'],source_binding=binding)
                for key in settings['profile_keys']:
                    relative=Path('analysis/frames')/key
                    for name in ('report.json','selected_frame.json','classified_frame.json','integrated_frame.json'):
                        copy(origin/relative/name,target/'frames'/key/name,entries[str(relative/name)]['sha256'])
                    frame_report=json.loads((work/target/'frames'/key/'report.json').read_text())
                    detail=frame_report['integration_details_archive'];assert detail in (None,'integration_details.tar')
                    if detail:copy(origin/relative/detail,target/'frames'/key/detail,entries[str(relative/detail)]['sha256'])
            census=next(row for row in census_index['runs'] if row['source_rng_seed']==seed)
            origin=repo/census['archive'];copy(origin/'delivery.json',Path('census_delivery.json'),census['delivery_sha256'])
            receipt=json.loads((work/'census_delivery.json').read_text());assert receipt['verified'] and receipt['status']=='COMPLETE_PFC_HISTORY_PROFILE_CENSUS_COMPLETE'
            entries={row['path']:row for row in receipt['files']};pairs=[]
            for key in settings['profile_keys']:
                target=Path('source_inputs/census')/key
                for name in ('report.json','frame_assignments.npz'):
                    relative=Path('analysis')/key/name;copy(origin/relative,target/name,entries[str(relative)]['sha256'])
                census_report=json.loads((work/target/'report.json').read_text())
                with np.load(work/target/'frame_assignments.npz',allow_pickle=False) as data:arrays={name:data[name] for name in data.files}
                result=read_pair(history_settings,census_report,arrays,work/'source_inputs/coarse/frames'/key,
                    work/'source_inputs/fine/frames'/key,bindings,work/'analysis'/key)
                pairs.append(dict(key=key,statistics=result['statistics'],per_clock=result['per_clock'],paired_selected_physical_windows=result['paired_selected_physical_windows']))
            write_json(work/'analysis/report.json',dict(status='COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK',
                source_rng_seed=seed,original_source_blocks=1,paired_profiles=len(pairs),pairs=pairs,source_costs=source_costs,
                full_source_requests=settings['full_source_requests'],readback_group=settings.get('group_name'),
                readback_group_index=settings.get('group_index'),
                third_spike_probability_or_bound_qualified=False,source_history_population_uncertainty_established=False,**BOUNDARY))
        commands.append(dict(phase='profile_readback_'+mode,exit_code=0))
    elif task=='profile_history':
        from Libs.population_dynamics.pfc_profile_history import prepare_inputs, execute_staged
        if mode=='prepare':
            result=prepare_inputs(settings,repo,work,copy)
        else:
            detector=Path(os.environ['NEURONSIM_RESOURCE_DETECTOR_SOURCE'])
            copy(detector,Path('resource_detector.py'))
            phase('resources',[sys.executable,'-B','resource_detector.py','--output',str(work/'available_resources.json')])
            selected_mode=os.environ.get('NEURONSIM_PROFILE_HISTORY_MODE','pilot')
            assert selected_mode in ('pilot','full')
            task_index=int(os.environ.get('SLURM_ARRAY_TASK_ID',os.environ.get('NEURONSIM_PROFILE_HISTORY_INDEX','0')))
            result=execute_staged(settings,repo,work,prepared,pe,copy,task_index=task_index,pilot=selected_mode=='pilot')
        commands.append(dict(phase='profile_history_'+mode,status=result['status'],exit_code=0))
    elif task=='profile_census':
        from Libs.population_dynamics.pfc_history_profile_census import prepare, execute
        if mode=='prepare':
            result=prepare(settings,repo,work,copy)
        else:
            detector=Path(os.environ['NEURONSIM_RESOURCE_DETECTOR_SOURCE'])
            copy(detector,Path('resource_detector.py'))
            phase('resources',[sys.executable,'-B','resource_detector.py','--output',str(work/'available_resources.json')])
            seed=int(os.environ.get('NEURONSIM_PROFILE_SOURCE_SEED',str(95001+int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))))
            result=execute(settings,repo,work,prepared,pe,copy,seed)
        commands.append(dict(phase='profile_census_'+mode,status=result['status'],exit_code=0))
    elif task=='profile_summary':
        plan_path=Path(os.environ['NEURONSIM_PROFILE_SUMMARY_PLAN'])
        copy(plan_path,Path('readback_plan.json'),os.environ['NEURONSIM_PROFILE_SUMMARY_PLAN_SHA256'])
        plan=json.loads((work/'readback_plan.json').read_text())
        index=dict(config_sha256=os.environ['NEURONSIM_SUMMARY_CONFIG_SHA256'],runs=[])
        for planned in plan['runs']:
            origin=repo/planned['archive'];data=(origin/'delivery.json').read_bytes()
            receipt=json.loads(data)
            assert receipt['verified'] and receipt['status']=='COMPLETE_PFC_HISTORY_PROFILE_CENSUS_COMPLETE'
            index['runs'].append(dict(planned,delivery_sha256=hashlib.sha256(data).hexdigest()))
        (work/'run_index.json').write_text(json.dumps(index,indent=2)+'\n')
        reference=plan['reference_csv'];copy(repo/reference['path'],Path('reference.csv'),reference['sha256'])
        phase('profile_summary',[sys.executable,'-B','-m','Libs.population_dynamics.pfc_history_profile_summary',
            '--config',str(work/config),'--run-index',str(work/'run_index.json'),'--project-root',str(repo),
            '--reference-csv',str(work/'reference.csv'),'--output',str(work/'analysis')])
    elif task=='design_comparison':
        references=list(settings['accepted_designs'].items())
        if 'accepted_allocation' in settings:references.append(('allocation',settings['accepted_allocation']))
        if 'accepted_baseline_comparison' in settings:references.append(('baseline',settings['accepted_baseline_comparison']))
        for kind,reference in references:
            origin=repo/reference['archive'];target=Path('project')/reference['archive']
            copy(origin/'delivery.json',target/'delivery.json',reference['delivery_sha256'])
            delivery=json.loads((work/target/'delivery.json').read_text());assert delivery['verified']
            entries={row['path']:row for row in delivery['files']}
            table='analysis/paired_block_diagnostics.csv' if kind=='uniform' else 'analysis/block_design_comparison.csv' if kind=='baseline' else 'analysis/block_diagnostics.csv'
            for relative in ('report.json','analysis/report.json',table):copy(origin/relative,target/relative,entries[relative]['sha256'])
    elif task=='stratified_summary':
        prep=settings['preparation'];origin=repo/prep['archive'];target=Path('project')/prep['archive']
        copy(origin/'delivery.json',target/'delivery.json',prep['delivery_sha256'])
        producer=json.loads((work/target/'delivery.json').read_text());assert producer['verified'] and producer['status']=='COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE'
        pf={row['path']:row for row in producer['files']}
        producer_config='Libs/configs/neural_population_pfc_stratified_history.yaml'
        for filename in ('report.json',producer_config):copy(origin/filename,target/filename,pf[filename]['sha256'])
        producer_settings=yaml.safe_load((work/target/producer_config).read_text())
        for reference,filenames in ((producer_settings['accepted_census'],('source_index.json','analysis/report.json')),
                                    (settings['accepted_uniform_comparison'],('report.json','analysis/report.json'))):
            origin=repo/reference['archive'];target=Path('project')/reference['archive']
            copy(origin/'delivery.json',target/'delivery.json',reference['delivery_sha256'])
            bound=json.loads((work/target/'delivery.json').read_text());assert bound['verified']
            entries={row['path']:row for row in bound['files']}
            for filename in filenames:copy(origin/filename,target/filename,entries[filename]['sha256'])
        for seed in settings['required_source_seeds']:
            relative=settings['pilot_archives'].get(str(seed),settings['array_archive_prefix']+str(seed-95001))
            origin=repo/relative;target=Path('project')/relative
            if not (origin/'delivery.json').exists():continue
            copy(origin/'delivery.json',target/'delivery.json',settings['pilot_delivery_sha256'].get(str(seed)))
            block=json.loads((work/target/'delivery.json').read_text());entries={row['path']:row for row in block['files']}
            # Stage only completed receipt payloads required by this numerical
            # reader. An incomplete producer stays visible as a failed receipt.
            if block['status']!='COMPLETE_PFC_STRATIFIED_HISTORY_COMPLETE':
                if 'report.json' in entries:copy(origin/'report.json',target/'report.json',entries['report.json']['sha256'])
                continue
            for filename in ('report.json','stratified_block_config.yaml',producer_config,'census/report.json',
                             'census/frame_assignments.npz','validation/report.json','validation/paired_frame.json'):
                copy(origin/filename,target/filename,entries[filename]['sha256'])
            for clock in ('coarse','fine'):
                prefix='validation/'+clock+'/'
                for filename in ('report.json','selected_frame.json','integrated_frame.json'):
                    key=prefix+filename
                    if key in entries:copy(origin/key,target/key,entries[key]['sha256'])
                if prefix+'integrated_frame.json' in entries:
                    for row in json.loads((work/target/(prefix+'integrated_frame.json')).read_text()):
                        for field in ('integration_report','suppressed_reference_report'):
                            if field in row:
                                key=prefix+row[field];copy(origin/key,target/key,entries[key]['sha256'])
    elif task=='stratified_history':
        reference=settings['accepted_census'];origin=repo/reference['archive']
        copy(origin/'delivery.json',Path('census_delivery.json'),reference['delivery_sha256'])
        cd=json.loads((work/'census_delivery.json').read_text());assert cd['verified'] and cd['status']=='COMPLETE_PFC_HISTORY_TIMING_CENSUS_COMPLETE'
        ce={row['path']:row for row in cd['files']}
        for relative,target in (('source_index.json','source_index.json'),('analysis/report.json','census_report.json')):
            copy(origin/relative,Path(target),ce[relative]['sha256'])
        census_report=json.loads((work/'census_report.json').read_text())
        assert census_report['status']=='COMPLETE_FULL_FRAME_CENSUS' and census_report['completed_original_blocks']==32
        assert census_report['completed_physical_windows']==6400000 and not census_report['failures']
        uniform=settings['accepted_uniform_comparison'];uniform_origin=repo/uniform['archive']
        copy(uniform_origin/'delivery.json',Path('uniform_delivery.json'),uniform['delivery_sha256'])
        ud=json.loads((work/'uniform_delivery.json').read_text());assert ud['verified'] and ud['status']=='COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE'
        ue={row['path']:row for row in ud['files']}
        key='Libs/population_dynamics/pfc_first_passage_clock_comparison.py'
        assert digest(work/key)==ue[key]['sha256'], 'STRATIFIED_PAIRED_READER_CHANGED'
        key='Libs/population_dynamics/pfc_first_passage_history_summary.py'
        copy(uniform_origin/key,Path('uniform_variance_reader.py'),ue[key]['sha256'])
        current_tree=ast.parse((work/key).read_text());prior_tree=ast.parse((work/'uniform_variance_reader.py').read_text())
        # The603570foundation adds suppressed-reference receipt validation in
        # read_block. The stratified estimator reuses its unchanged variance
        # functions; it never calls that uniform-sample reader.
        for tree in (current_tree,prior_tree):
            tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!='read_block']
        assert ast.dump(current_tree)==ast.dump(prior_tree), 'STRATIFIED_VARIANCE_REUSE_CHANGED'
        copy(uniform_origin/'analysis/report.json',Path('uniform_report.json'),ue['analysis/report.json']['sha256'])
        assert json.loads((work/'uniform_report.json').read_text())['available_original_block_pairs']==32
        index=json.loads((work/'source_index.json').read_text())
        assert list(index['source_blocks'])==[str(seed) for seed in settings['required_source_seeds']]==[str(seed) for seed in range(95001,95033)]
        if mode=='prepare':
            (work/'analysis').mkdir()
            (work/'analysis/report.json').write_text(json.dumps(dict(status='PREPARED_FRESH_PAIRED_STRATIFIED_HISTORY',required_original_blocks=32,nominal_physical_samples_per_block=settings['sample_size'],inner_samples_per_eligible_clock_window=settings['importance_sampling']['samples'],nominal_inner_trial_budget_per_clock=settings['sample_size']*settings['importance_sampling']['samples'],integration_seed_namespace=settings['integration_seed_namespace'],source_reference_reconstructions_run=0,source_events_read=False,scientific_fidelity_qualified=False,confirmation_execution_released=False),indent=2)+'\n')
        else:
            block_index=int(os.environ.get('SLURM_ARRAY_TASK_ID',os.environ.get('NEURONSIM_STRATIFIED_BLOCK_INDEX','0')))
            seed=settings['required_source_seeds'][block_index]
            for filename in ('report.json','frame_assignments.npz'):
                key='analysis/'+str(seed)+'/'+filename
                copy(origin/key,Path('census')/filename,ce[key]['sha256'])
            block=index['source_blocks'][str(seed)]
            pair_bytes=sum(source['raw']['bytes'] for source in block.values())
            assert shutil.disk_usage(work).free>=pair_bytes+1024**3, 'STRATIFIED_PAIR_SCRATCH_CAPACITY'
            resolved=dict(settings,source_rng_seed=seed,case_id=f'pfc_{seed}_strong_cue',sampling_seed_components=[20261009,seed,5,settings['integration_seed_namespace']])
            resolved['clocks']={clock:dict(spec) for clock,spec in settings['clocks'].items()}
            for clock,source in block.items():
                source_origin=repo/source['archive'];target=Path('source_inputs')/clock
                copy(source_origin/'delivery.json',target/'delivery.json',source['delivery_sha256'])
                sd=json.loads((work/target/'delivery.json').read_text())
                if clock=='coarse':assert sd['source_execution_complete'] and sd['exit_code']==0 and sd['status']=='ARCHIVED_PFC_SOURCE_EXECUTION'
                else:assert sd['verified'] and sd['status'] in ('COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT','COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK')
                se={row['path']:row for row in sd['files']};case=Path(source['case_path'])
                assert se[str(case/'raw.npz')]==source['raw']
                for filename in ('config.yaml','report.json','initial_voltage.npy','raw.npz'):
                    key=str(case/filename);assert (source_origin/key).stat().st_size==se[key]['bytes']
                    copy(source_origin/key,target/filename,se[key]['sha256'])
                sr=json.loads((work/target/'report.json').read_text())
                assert sr['source_execution_complete'] and sr['status']==('PFC_SOURCE_EXECUTION_COMPLETE' if clock=='coarse' else 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE')
                resolved['clocks'][clock]['source']=dict(archive=source['archive'],case_path=source['case_path'],delivery_sha256=source['delivery_sha256'])
            (work/'stratified_block_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False))
            phase('paired_stratified_history',[sys.executable,'-B','-m','Libs.population_dynamics.pfc_stratified_history',
                '--config',str(work/'stratified_block_config.yaml'),'--census',str(work/'census'),
                '--coarse',str(work/'source_inputs/coarse'),'--fine',str(work/'source_inputs/fine'),'--output',str(work/'validation')])
    elif task=='timing_census':
        reference=settings['accepted_timing_check'];origin=repo/reference['archive']
        copy(origin/'delivery.json',Path('timing_check_delivery.json'),reference['delivery_sha256'])
        td=json.loads((work/'timing_check_delivery.json').read_text())
        assert td['verified'] and td['status']=='COMPLETE_PFC_HISTORY_TIMING_PREDICTION_PREPARE'
        te={row['path']:row for row in td['files']}
        copy(origin/'analysis/report.json',Path('timing_check_report.json'),te['analysis/report.json']['sha256'])
        assert json.loads((work/'timing_check_report.json').read_text())['decision']['prepare_timing_stratum_census'] is True
        if mode=='prepare':
            source_blocks={}
            for clock,spec in settings['clocks'].items():
                copy(repo/spec['registry'],Path(spec['registry']),spec['registry_sha256'])
                registry=yaml.safe_load((work/spec['registry']).read_text())
                rows=registry['blocks'] if clock=='coarse' else registry['source_blocks']
                assert [row['source_rng_seed'] for row in rows]==settings['required_source_seeds']==list(range(95001,95033))
                for row in rows:
                    seed=row['source_rng_seed'];source=row['coarse_references']['strong_cue'] if clock=='coarse' else row['source']
                    origin=repo/source['archive'];target=Path('source_metadata')/str(seed)/clock
                    copy(origin/'delivery.json',target/'delivery.json',source['delivery_sha256'])
                    delivery=json.loads((work/target/'delivery.json').read_text())
                    if clock=='coarse':
                        assert delivery['status']=='ARCHIVED_PFC_SOURCE_EXECUTION' and delivery['source_execution_complete'] and delivery['exit_code']==0
                    else:
                        assert delivery['verified'] and delivery['status'] in ('COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT','COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK')
                    entries={entry['path']:entry for entry in delivery['files']}
                    case=Path('source_run') if clock=='coarse' else Path(source['case_path'])
                    assert entries[str(case/'report.json')]['sha256']==source['report_sha256']
                    for filename in ('report.json','config.yaml'):
                        copy(origin/case/filename,target/filename,entries[str(case/filename)]['sha256'])
                    report=json.loads((work/target/'report.json').read_text())
                    assert report['source_execution_complete'] and report['status']==('PFC_SOURCE_EXECUTION_COMPLETE' if clock=='coarse' else 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE')
                    assert report['selected_case']['source_rng_seed']==seed and report['selected_case']['case_id']==f'pfc_{seed}_strong_cue'
                    assert report['resolved_config']==yaml.safe_load((work/target/'config.yaml').read_text())
                    raw=entries[str(case/'raw.npz')]
                    source_blocks.setdefault(str(seed),{})[clock]=dict(archive=source['archive'],delivery_sha256=source['delivery_sha256'],case_path=str(case),metadata=str(target),raw=raw)
            index=dict(required_original_blocks=32,source_blocks=source_blocks,
                       raw_bytes=sum(v['raw']['bytes'] for block in source_blocks.values() for v in block.values()),
                       max_pair_bytes=max(sum(v['raw']['bytes'] for v in block.values()) for block in source_blocks.values()))
            (work/'source_index.json').write_text(json.dumps(index,indent=2)+'\n')
            (work/'analysis').mkdir()
            (work/'analysis/report.json').write_text(json.dumps(dict(status='PREPARED_FULL_FRAME_CENSUS_INPUTS',required_original_blocks=32,completed_original_blocks=0,raw_bytes_declared=index['raw_bytes'],max_pair_bytes_declared=index['max_pair_bytes'],source_events_read=False,scientific_fidelity_qualified=False,confirmation_execution_released=False),indent=2)+'\n')
        else:
            copy(prepared/'source_index.json',Path('source_index.json'),pe['source_index.json']['sha256'])
            index=json.loads((work/'source_index.json').read_text())
            assert list(index['source_blocks'])==[str(seed) for seed in settings['required_source_seeds']]
            required_scratch=index['max_pair_bytes']+1024**3
            assert shutil.disk_usage(work).free>=required_scratch, 'CENSUS_PAIR_SCRATCH_CAPACITY'
            from Libs.population_dynamics.pfc_history_timing_census import run_pair
            analysis=work/'analysis';analysis.mkdir();completed=[];failures=[]
            for seed in settings['required_source_seeds']:
                start=time.perf_counter();pair_dir=work/'project'/str(seed)
                try:
                    block=index['source_blocks'][str(seed)]
                    for clock,source in block.items():
                        for filename in ('report.json','config.yaml'):
                            relative=Path(source['metadata'])/filename
                            copy(prepared/relative,Path('project')/str(seed)/clock/filename,pe[str(relative)]['sha256'])
                        raw=source['raw'];assert shutil.disk_usage(work).free>=raw['bytes']+1024**3, 'CENSUS_PAIR_SCRATCH_CAPACITY'
                        origin=repo/source['archive']/raw['path'];assert origin.stat().st_size==raw['bytes']
                        copy(origin,Path('project')/str(seed)/clock/'raw.npz',raw['sha256'])
                    block_settings=dict(settings,source_rng_seed=seed,case_id=f'pfc_{seed}_strong_cue')
                    result=run_pair(block_settings,pair_dir/'coarse',pair_dir/'fine',analysis/str(seed))
                    completed.append(result)
                    commands.append(dict(phase='census',source_rng_seed=seed,exit_code=0,wall_seconds=time.perf_counter()-start,frame_size=result['frame_size']))
                except Exception as exc:
                    failures.append(dict(source_rng_seed=seed,type=type(exc).__name__,message=str(exc)))
                    commands.append(dict(phase='census',source_rng_seed=seed,exit_code=1,wall_seconds=time.perf_counter()-start))
                    traceback.print_exc()
                finally:
                    if pair_dir.exists():shutil.rmtree(pair_dir)
                print(json.dumps(commands[-1]),flush=True)
            counts={str(i):sum(row['joint_time_bin_counts'][str(i)] for row in completed) for i in range(16)}
            frame_size=sum(row['frame_size'] for row in completed)
            aggregate=dict(status='COMPLETE_FULL_FRAME_CENSUS' if len(completed)==32 and not failures else 'INCOMPLETE_FULL_FRAME_CENSUS',
                required_original_blocks=32,completed_original_blocks=len(completed),required_physical_windows=6400000,
                completed_physical_windows=frame_size,failures=failures,completed_source_seeds=[row['source_rng_seed'] for row in completed],
                joint_time_bin_counts=counts,joint_time_bin_weights=({key:value/frame_size for key,value in counts.items()} if len(completed)==32 and not failures else None),
                coarse_zero_fine_eligible_windows=sum(row['coarse_zero_fine_eligible_windows'] for row in completed),
                fine_zero_coarse_eligible_windows=sum(row['fine_zero_coarse_eligible_windows'] for row in completed),
                multispike_windows={clock:sum(row['clocks'][clock]['multispike_windows'] for row in completed) for clock in ('coarse','fine')},
                raw_bytes_declared=index['raw_bytes'],staging='one source pair at a time in node-local scratch',
                weight_scope='Exact conditional archived-frame frequencies; not independent windows or source-law probabilities',
                population_confidence_interval_computed=False,future_stratified_precision_established=False,
                new_neural_trajectories=0,new_independent_source_blocks=0,scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0)
            (analysis/'report.json').write_text(json.dumps(aggregate,indent=2)+'\n')
            with (analysis/'block_census.csv').open('w',newline='') as stream:
                fields=['source_rng_seed','frame_size','coarse_zero_fine_eligible_windows','fine_zero_coarse_eligible_windows']
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                writer.writerows({key:row[key] for key in fields} for row in completed)
            assert not failures and len(completed)==32 and frame_size==6400000, 'ALL32_CENSUS_PAIRS_REQUIRED'
    elif task=='timing_prediction':
        reference=settings['accepted_coarse_summary'];origin=repo/reference['archive']
        target=Path('project')/reference['archive']
        copy(origin/'delivery.json',target/'delivery.json',reference['delivery_sha256'])
        accepted=json.loads((work/target/'delivery.json').read_text())
        assert accepted['verified'] and accepted['status']=='COMPLETE_PFC_HISTORY_SUMMARY_COMPLETE'
        files={row['path']:row for row in accepted['files']}
        for name_in_archive in ('report.json','analysis/report.json','analysis/block_diagnostics.csv',
                                'Libs/configs/neural_population_pfc_first_passage_history_summary.yaml'):
            copy(origin/name_in_archive,target/name_in_archive,files[name_in_archive]['sha256'])
        accepted_report=json.loads((work/target/'report.json').read_text())
        bound_inputs={row['path']:row['sha256'] for row in accepted_report['inputs']}
        with (work/target/'analysis/block_diagnostics.csv').open() as stream:
            blocks=list(csv.DictReader(stream))
        assert [int(row['source_rng_seed']) for row in blocks]==settings['required_source_seeds']
        for block in blocks:
            relative=Path(block['archive'])/'validation/integrated_frame.json'
            copy(repo/relative,Path('project')/relative,bound_inputs[str(Path('project')/relative)])
    else:
        clocks={'coarse':settings} if task=='history' else settings['clocks']
        for clock,spec in clocks.items():
            registry=spec['source_registry'];copy(repo/registry['path'],Path('project')/registry['path'],registry['sha256'])
            for seed in settings['required_source_seeds']:
                relative=spec['pilot_archives'].get(str(seed),spec['array_archive_prefix']+str(seed-95001))
                origin=repo/relative
                if not (origin/'delivery.json').exists():continue
                target=Path('project')/relative
                copy(origin/'delivery.json',target/'delivery.json',spec['pilot_delivery_sha256'].get(str(seed)))
                manifest=json.loads((work/target/'delivery.json').read_text());entries={r['path']:r for r in manifest['files']}
                for name_in_archive in ('report.json','validation/report.json','validation/selected_frame.json','validation/integrated_frame.json'):
                    copy(origin/name_in_archive,target/name_in_archive,entries[name_in_archive]['sha256'])
                rows=json.loads((work/target/'validation/integrated_frame.json').read_text())
                for row in rows:
                    if row.get('category') in ('conditional_integration_required','suppressed_network_reference_required'):
                        name_in_archive='validation/'+row['integration_report']
                        copy(origin/name_in_archive,target/name_in_archive,entries[name_in_archive]['sha256'])
                        if row.get('suppressed_reference_resolved'):
                            name_in_archive='validation/'+row['suppressed_reference_report']
                            copy(origin/name_in_archive,target/name_in_archive,entries[name_in_archive]['sha256'])
    if task not in ('timing_census','stratified_history','profile_census','profile_summary','profile_history','profile_readback','profile_coverage'):
        args=[sys.executable,'-B','-m','Libs.population_dynamics.'+module.stem,
              '--config',str(work/config),'--project-root',str(work/'project'),'--output',str(work/'analysis')]
        if mode=='prepare' and task not in ('timing_prediction','design_comparison'):args.append('--allow-partial')
        phase('readback',args)
    status='COMPLETE_'+status_name+'_'+mode.upper();exit_code=0
except Exception as exc:
    error=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
finally:
    suites=[]
    if (work/'tests.xml').exists():suites=[dict(n.attrib) for n in ET.parse(work/'tests.xml').getroot().iter('testsuite')]
    report=dict(status=status,exit_code=exit_code,error=error,mode=mode,task=task,job_id=job,node=socket.gethostname(),config_source=str(config_source),inputs=inputs,
        commands=commands,test_suites=suites,slurm_step_id=os.environ.get('SLURM_STEP_ID'),wall_seconds=time.perf_counter()-begin,finished_utc=datetime.now(timezone.utc).isoformat(),
        scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,new_independent_source_blocks=0)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        archive.mkdir(parents=True,exist_ok=False);files=[]
        for path in sorted(work.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(work)
            if relative.parts[0] in ('project','scratch','test_tmp','source_inputs') or '__pycache__' in relative.parts:continue
            destination=archive/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            sha=digest(path);assert digest(destination)==sha
            files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=sha))
        (archive/'delivery.json').write_text(json.dumps(dict(status=status,job_id=job,verified=True,files=files),indent=2)+'\n')
        delivered=True;print(json.dumps(dict(status=status,archive=str(archive),exit_code=exit_code)),flush=True)
    finally:
        if delivered:shutil.rmtree(work)
        else:print('COPYBACK_FAILED_SCRATCH_PRESERVED:'+str(work),file=sys.stderr)
raise SystemExit(exit_code)
PY_SUMMARY

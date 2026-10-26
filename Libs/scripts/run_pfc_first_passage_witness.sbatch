#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_FirstPassageWitness
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-first-passage-witness-%j.out
#SBATCH --error=/tmp/neuronsim-first-passage-witness-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_WITNESS_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_WITNESS'
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
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

task = os.environ.get('NEURONSIM_PASSAGE_TASK', 'witness')
assert task in ('witness', 'history', 'fine_history', 'suppression', 'history_suppression', 'fine_suppression')
stem = 'pfc_first_passage_'+('suppression' if task == 'fine_suppression' else ('history' if task in ('fine_history','history_suppression') else task))
status_name = 'PFC_FIRST_PASSAGE_'+task.upper()
expected_tests = {'fine_suppression':28, 'history':15, 'witness':13, 'suppression':15}.get(task,7)
mode = os.environ.get('NEURONSIM_WITNESS_MODE', 'prepare')
assert mode in ('prepare', 'execute')
assert task != 'history_suppression' or mode == 'prepare'
assert socket.gethostname().split('.')[0] == ('cpu04' if mode == 'prepare' else 'cpu03')
assert not os.environ.get('SLURM_JOB_GPUS')
repo = Path(os.environ['NEURONSIM_PROJECT_ROOT'])
assert (repo/'PROJECT_PREFIX').read_text().strip() == 'NrnSim'
job = os.environ['SLURM_JOB_ID']
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    job = os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
family = Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement')/('first_passage_'+task)
archive = repo/family/mode/('run_'+job)
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix='neuronsim-first-passage-witness-'+job+'.', dir='/tmp'))
(work/'scratch').mkdir()
env = dict(os.environ, PYTHONPATH=str(work), TMPDIR=str(work/'scratch'))
started = time.perf_counter(); inputs = []; commands = []; delivered = False; exit_code = 1; error = None
status = 'FAILED_'+status_name+'_'+mode.upper()
config = Path('Libs/configs')/('neural_population_pfc_first_passage_'+task+'.yaml')
config_source = Path(os.environ.get('NEURONSIM_WITNESS_CONFIG_SOURCE',str(config)))
if not config_source.is_absolute(): config_source = repo/config_source
module = Path('Libs/population_dynamics')/(stem+'.py')
test = Path('Libs/population_dynamics')/('test_pfc_first_passage_'+task+'.py')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def copy(source, relative, expected=None):
    target = work/relative; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target); actual = digest(target)
    if expected is not None and actual != expected:
        raise ValueError('FIRST_PASSAGE_WITNESS_INPUT_CHANGED:'+str(relative))
    inputs.append(dict(source=str(source), path=str(relative), sha256=actual))

def phase(name, args):
    begin = time.perf_counter()
    with (work/(name+'.stdout')).open('w') as out, (work/(name+'.stderr')).open('w') as err:
        result = subprocess.run(args, cwd=work, env=env, stdout=out, stderr=err)
    commands.append(dict(phase=name, command=args, exit_code=result.returncode, wall_seconds=time.perf_counter()-begin))
    print(json.dumps(commands[-1]), flush=True)
    if result.returncode: raise RuntimeError('FIRST_PASSAGE_WITNESS_'+name.upper()+'_FAILED:'+str(result.returncode))

def fine_suppression_flow():
    """Reuse frozen profile software and two exact saved history payloads."""
    import tarfile
    settings = yaml.safe_load((work/config).read_text())
    assert settings['schema'] == 'pfc_fine_clock_suppression_v1'
    assert [row['dt_ms'] for row in settings['cases']] == [.025, .0125]
    bound = repo/settings['software_preparation']['archive']
    copy(bound/'delivery.json', Path('software_delivery.json'), settings['software_preparation']['delivery_sha256'])
    software = json.loads((work/'software_delivery.json').read_text())
    assert software['verified'] and software['status'] == 'COMPLETE_PFC_PROFILE_HISTORY_PREPARE'
    se = {row['path']:row for row in software['files']}
    for key, entry in se.items():
        if key.startswith('Libs/') and key.endswith('.py'):
            copy(bound/key, Path(key), entry['sha256'])
    oracle = repo/settings['source_oracle']['archive']
    copy(oracle/'delivery.json', Path('oracle_delivery.json'), settings['source_oracle']['delivery_sha256'])
    od = json.loads((work/'oracle_delivery.json').read_text())
    assert od['verified'] and od['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_FINE_HISTORY_PREPARE'
    oe = {row['path']:row for row in od['files']}
    key = 'Libs/population_dynamics/pfc_deterministic_replay.py'
    copy(oracle/key, Path('source_oracle.py'), oe[key]['sha256'])
    env['NEURONSIM_SUPPRESSION_BASELINE'] = str(work/'source_oracle.py')
    implementation = [Path('Libs/population_dynamics')/(name+'.py') for name in
        ('pfc_first_passage_witness', 'pfc_first_passage_suppression',
         'test_pfc_first_passage_witness', 'test_pfc_first_passage_suppression')]
    if mode == 'prepare':
        for key in implementation: copy(repo/key, key)
        phase('tests', [sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
            *[str(key) for key in implementation if key.name.startswith('test_')],
            '--basetemp='+str(work/'test_tmp'), '--junitxml='+str(work/'tests.xml')])
    else:
        preparation = Path(os.environ['NEURONSIM_WITNESS_PREPARATION'])
        copy(preparation/'delivery.json', Path('preparation_delivery.json'),
             os.environ['NEURONSIM_WITNESS_PREPARATION_SHA256'])
        pd = json.loads((work/'preparation_delivery.json').read_text())
        assert pd['verified'] and pd['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_FINE_SUPPRESSION_PREPARE'
        pe = {row['path']:row for row in pd['files']}
        assert digest(work/config) == pe[str(config)]['sha256']
        for key in implementation: copy(preparation/key, key, pe[str(key)]['sha256'])
        for key in se:
            if key.startswith('Libs/') and key.endswith('.py') and Path(key) not in implementation:
                assert digest(work/key) == pe[key]['sha256']
        copy(preparation/'report.json', Path('preparation_report.json'), pe['report.json']['sha256'])
        pr = json.loads((work/'preparation_report.json').read_text())
        assert pr['exit_code'] == 0
        assert sum(int(row['tests']) for row in pr['test_suites']) == expected_tests
        assert all(int(row['failures']) == int(row['errors']) == int(row['skipped']) == 0
                   for row in pr['test_suites'])
        commands.append(dict(phase='fine_clock_tests_reused',tests=expected_tests,exit_code=0))
    for row in settings['cases']:
        target = Path('source_inputs')/row['name']
        origin = repo/row['source']['archive']
        copy(origin/'delivery.json', target/'source_delivery.json', row['source']['delivery_sha256'])
        sd = json.loads((work/target/'source_delivery.json').read_text())
        assert sd['verified']
        entries = {entry['path']:entry for entry in sd['files']}
        case_path = row['source']['case_path']
        names = ['config.yaml','report.json','initial_voltage.npy']
        if mode == 'execute': names.append('raw.npz')
        for name in names:
            key = case_path+'/'+name
            copy(origin/key,target/'source'/name,entries[key]['sha256'])
        assert digest(work/target/'source/report.json') == row['source']['report_sha256']
        sr = json.loads((work/target/'source/report.json').read_text())
        assert sr['status'] == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE' and sr['source_execution_complete']
        assert sr['refinement_seeds'] == row['source']['refinement_seeds']
        history = repo/row['history']['archive']
        copy(history/'delivery.json',target/'history_delivery.json',row['history']['delivery_sha256'])
        hd = json.loads((work/target/'history_delivery.json').read_text())
        assert hd['verified'] and hd['status'] == 'COMPLETE_PFC_PROFILE_HISTORY_COMPLETE'
        he = {entry['path']:entry for entry in hd['files']}
        key = row['history']['frame_path']+'/integrated_frame.json'
        copy(history/key,target/'integrated_frame.json',he[key]['sha256'])
        frame = json.loads((work/target/'integrated_frame.json').read_text())[row['sample_index']]
        assert frame['sample_index'] == row['sample_index'] and frame['proposal_hits'] > 0
        (work/target/'frame_row.json').write_text(json.dumps(frame,indent=2)+'\n')
        if mode == 'execute':
            key = row['history']['details_archive']
            copy(history/key, target/'integration_details.tar', he[key]['sha256'])
            prefix = 'sample_'+str(row['sample_index']).zfill(3)
            members = {prefix+'/reference/conditional_initial.json':'reference/conditional_initial.json',
                prefix+'/reference/reference_window.npz':'reference/reference_window.npz',
                prefix+'/integration/first_hit_witnesses.npz':'first_hit_witnesses.npz'}
            with tarfile.open(work/target/'integration_details.tar','r:') as stream:
                for name, relative in members.items():
                    member = stream.getmember(name); assert member.isfile()
                    destination = work/target/relative; destination.parent.mkdir(parents=True,exist_ok=True)
                    with stream.extractfile(member) as source, destination.open('wb') as sink:
                        shutil.copyfileobj(source,sink)
            (work/target/'integration_details.tar').unlink()
    copy(Path(os.environ['NEURONSIM_WITNESS_LAUNCHER']), Path('launcher.sbatch'))
    sys.path.insert(0,str(work))
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work/'resource_preflight.json', 4)
    assert health['status'] == 'PASS', health['reasons']
    if mode == 'execute':
        phase('fine_clock_suppression',[sys.executable,'-B','-m','Libs.population_dynamics.pfc_first_passage_suppression',
            '--config',str(work/config),'--inputs',str(work/'source_inputs'),'--output',str(work/'validation')])

try:
    copy(config_source, config, os.environ['NEURONSIM_WITNESS_CONFIG_SHA256'])
    if task == 'fine_suppression':
        fine_suppression_flow()
    else:
        declared_settings = yaml.safe_load((work/config).read_text())
        settings = deepcopy(declared_settings)
        assert settings['schema'] == stem+'_v1'
        if task == 'fine_history' and ('NEURONSIM_HISTORY_BLOCK_INDEX' in os.environ or 'SLURM_ARRAY_TASK_ID' in os.environ):
            index = int(os.environ['SLURM_ARRAY_TASK_ID'] if 'SLURM_ARRAY_TASK_ID' in os.environ
                        else os.environ['NEURONSIM_HISTORY_BLOCK_INDEX'])
            blocks = settings['source_blocks']
            assert [b['source_rng_seed'] for b in blocks] == list(range(95001,95033)) and 1 <= index <= 31
            block = blocks[index]; seed = block['source_rng_seed']
            settings['source'] = block['source']
            settings['source_rng_seed'] = seed
            settings['case_id'] = 'pfc_'+str(seed)+'_strong_cue'
            settings['sampling_seed_components'][1] = seed
            pairs = settings['paired_coarse_runs']
            paired_archive = pairs['pilot_archives'].get(str(seed),pairs['array_archive_prefix']+str(index))
            paired_delivery = repo/paired_archive/'delivery.json'
            pair_status = json.loads(paired_delivery.read_text())
            assert pair_status['verified'] and pair_status['status'] in ('COMPLETE_PFC_FIRST_PASSAGE_HISTORY_EXECUTE','COMPLETE_PFC_FIRST_PASSAGE_FINE_HISTORY_EXECUTE')
            settings['paired_coarse'] = dict(archive=paired_archive,delivery_sha256=digest(paired_delivery))
            settings['resolved_source_registry_block_index'] = index
        importance = repo/settings['importance']['archive']
        copy(importance/'delivery.json', Path('importance_delivery.json'), settings['importance']['delivery_sha256'])
        delivery = json.loads((work/'importance_delivery.json').read_text())
        assert delivery['verified'] and delivery['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_IMPORTANCE'
        entries = {row['path']: row for row in delivery['files']}
        for name, entry in entries.items():
            relative = Path(name)
            if relative.parts[0] == 'Libs' and relative.suffix == '.py':
                copy(importance/relative, relative, entry['sha256'])
            elif name in ('source_config.yaml','reference/conditional_initial.json','reference/reference_window.npz','reference/report.json'):
                copy(importance/relative, relative, entry['sha256'])
        for name, target in (('report.json','importance_report.json'),
                             ('analysis/report.json','importance_analysis_report.json'),
                             ('analysis/first_hit_witnesses.npz','first_hit_witnesses.npz')):
            copy(importance/name, Path(target), entries[name]['sha256'])
        prior = json.loads((work/'importance_report.json').read_text())
        assert prior['exit_code'] == 0
        if task == 'fine_history':
            fine_origin = repo/settings['source']['archive']
            copy(fine_origin/'delivery.json', Path('fine_source_delivery.json'), settings['source']['delivery_sha256'])
            fd = json.loads((work/'fine_source_delivery.json').read_text())
            assert fd['verified']
            fe = {row['path']:row for row in fd['files']}
            report_key = settings['source']['case_path']+'/report.json'
            copy(fine_origin/report_key, Path('fine_source_report.json'), settings['source']['report_sha256'])
            assert fe[report_key]['sha256'] == settings['source']['report_sha256']
            fine_report = json.loads((work/'fine_source_report.json').read_text())
            assert fine_report['status'] == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE' and fine_report['source_execution_complete']
            for name in ('pfc_coupled_drive','pfc_clock_coupling'):
                key = 'Libs/population_dynamics/'+name+'.py'
                copy(fine_origin/key, Path(key), fe[key]['sha256'])
            for name in ('pfc_coupled_drive','pfc_clock_coupling','pfc_deterministic_replay','pfc_native_drive'):
                assert digest(work/('Libs/population_dynamics/'+name+'.py')) == fine_report['code'][name]
            paired = repo/settings['paired_coarse']['archive']
            copy(paired/'delivery.json', Path('paired_coarse_delivery.json'), settings['paired_coarse']['delivery_sha256'])
            pair_delivery = json.loads((work/'paired_coarse_delivery.json').read_text())
            assert pair_delivery['verified'] and pair_delivery['status'] in ('COMPLETE_PFC_FIRST_PASSAGE_HISTORY_EXECUTE','COMPLETE_PFC_FIRST_PASSAGE_FINE_HISTORY_EXECUTE')
            pair_entries = {row['path']:row for row in pair_delivery['files']}
            copy(paired/'validation/selected_frame.json', Path('paired_coarse_frame.json'), pair_entries['validation/selected_frame.json']['sha256'])
        implementation = [module, test]
        tests_to_run = [test]
        if task == 'suppression':
            source_module = Path('Libs/population_dynamics/pfc_deterministic_replay.py')
            copy(work/source_module, Path('source_oracle.py'), entries[str(source_module)]['sha256'])
            env['NEURONSIM_SUPPRESSION_BASELINE'] = str(work/'source_oracle.py')
            implementation.extend([source_module,Path('Libs/population_dynamics/pfc_first_passage_witness.py')])
            witness_origin = repo/settings['witness_source']['archive']
            copy(witness_origin/'delivery.json', Path('witness_delivery.json'), settings['witness_source']['delivery_sha256'])
            wd = json.loads((work/'witness_delivery.json').read_text())
            assert wd['verified'] and wd['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_WITNESS_EXECUTE'
            we = {row['path']:row for row in wd['files']}
            key = 'Libs/population_dynamics/pfc_first_passage_witness.py'
            copy(witness_origin/key, Path(key), we[key]['sha256'])
            assert we[str(source_module)]['sha256'] == digest(work/'source_oracle.py')
            if mode == 'execute':
                for name in ('report.json','source_raw.npz','native_prefix_rng.npz'):
                    key = 'validation/'+name
                    copy(witness_origin/key, Path('witness_inputs')/name, we[key]['sha256'])
        if task == 'history_suppression':
            import ast
            qualified = repo/settings['suppression_preparation']['archive']
            copy(qualified/'delivery.json', Path('suppression_preparation_delivery.json'),
                 settings['suppression_preparation']['delivery_sha256'])
            qd = json.loads((work/'suppression_preparation_delivery.json').read_text())
            assert qd['verified'] and qd['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_SUPPRESSION_PREPARE'
            qe = {row['path']:row for row in qd['files']}
            copy(qualified/'report.json', Path('suppression_preparation_report.json'), qe['report.json']['sha256'])
            qr = json.loads((work/'suppression_preparation_report.json').read_text())
            assert qr['exit_code'] == 0 and sum(int(row['tests']) for row in qr['test_suites']) == 7
            assert all(int(row['failures']) == int(row['errors']) == int(row['skipped']) == 0 for row in qr['test_suites'])
            for name in ('pfc_deterministic_replay','pfc_first_passage_witness','test_pfc_first_passage_suppression'):
                key = 'Libs/population_dynamics/'+name+'.py'
                copy(qualified/key,Path(key),qe[key]['sha256'])
            source_key = 'Libs/population_dynamics/pfc_deterministic_replay.py'
            assert digest(repo/source_key) == qe[source_key]['sha256']
            # Reuse the seven suppression checks only when the new adapter is the
            # sole AST addition to the previously accepted suppression module.
            helper = Path('Libs/population_dynamics/pfc_first_passage_suppression.py')
            copy(qualified/helper,Path('qualified_suppression.py'),qe[str(helper)]['sha256'])
            old_tree = ast.parse((work/'qualified_suppression.py').read_text())
            current_tree = ast.parse((repo/helper).read_text())
            adapters = [node for node in current_tree.body if isinstance(node,ast.FunctionDef) and node.name == 'replay_history_window']
            assert len(adapters) == 1
            current_tree.body.remove(adapters[0])
            assert ast.dump(current_tree,include_attributes=False) == ast.dump(old_tree,include_attributes=False)
            baseline_test = Path('Libs/population_dynamics/test_pfc_first_passage_history.py')
            implementation.extend([helper,baseline_test,Path('Libs/population_dynamics/pfc_first_passage_history_summary.py')])
            tests_to_run.append(baseline_test)
            commands.append(dict(phase='suppression_kernel_tests_reused',tests=7,archive=str(qualified),exit_code=0))
        if mode == 'prepare':
            for relative in implementation: copy(repo/relative, relative)
        else:
            preparation = Path(os.environ['NEURONSIM_WITNESS_PREPARATION'])
            copy(preparation/'delivery.json', Path('preparation_delivery.json'), os.environ['NEURONSIM_WITNESS_PREPARATION_SHA256'])
            pd = json.loads((work/'preparation_delivery.json').read_text())
            assert pd['verified'] and pd['status'] == 'COMPLETE_'+status_name+'_PREPARE'
            pe = {row['path']:row for row in pd['files']}
            if digest(work/config) != pe[str(config)]['sha256']:
                # The registered block extension changes only source provenance.
                # Reuse the exact tested extractor and fixed pilot sampling budget.
                allowed = ('source_registry',) if task == 'history' else ('source_blocks','paired_coarse_runs')
                assert task in ('history','fine_history') and all(k in declared_settings for k in allowed)
                copy(preparation/config, Path('prepared_analysis_config.yaml'), pe[str(config)]['sha256'])
                previous_settings = yaml.safe_load((work/'prepared_analysis_config.yaml').read_text())
                assert {k:v for k,v in declared_settings.items() if k not in allowed} == previous_settings
            for relative in implementation: copy(preparation/relative, relative, pe[str(relative)]['sha256'])
            if task == 'fine_history':
                for name in ('pfc_coupled_drive','pfc_clock_coupling','pfc_deterministic_replay','pfc_native_drive'):
                    key = 'Libs/population_dynamics/'+name+'.py'
                    assert digest(work/key) == pe[key]['sha256']
            copy(preparation/'report.json', Path('preparation_report.json'), pe['report.json']['sha256'])
            prepared = json.loads((work/'preparation_report.json').read_text())
            assert prepared['exit_code'] == 0
            suites = prepared['test_suites']
            prior_test_count = sum(int(row['tests']) for row in suites)
            if task == 'history' and prior_test_count in (3, 15):
                expected_tests = prior_test_count
            if task in ('witness','suppression') and prior_test_count in (7,13,11,15):
                expected_tests = prior_test_count
            assert prior_test_count == expected_tests
            assert all(int(row['failures']) == int(row['errors']) == int(row['skipped']) == 0 for row in suites)
            commands.append(dict(phase='adapter_tests_reused', archive=str(preparation), tests=expected_tests, exit_code=0))
        copy(Path(os.environ['NEURONSIM_WITNESS_LAUNCHER']), Path('launcher.sbatch'))
        sys.path.insert(0, str(work))
        from Libs.population_dynamics.resource_probe import probe
        health = probe(work/'resource_preflight.json', 4)
        assert health['status'] == 'PASS', health['reasons']
        if mode == 'prepare':
            phase('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*[str(path) for path in tests_to_run],
                           '--basetemp='+str(work/'test_tmp'),'--junitxml='+str(work/'tests.xml')])
        else:
            analysis_config = work/config
            if task == 'fine_history' and settings != declared_settings:
                analysis_config = work/'resolved_analysis_config.yaml'
                analysis_config.write_text(yaml.safe_dump(settings,sort_keys=False))
            if task == 'history' and ('NEURONSIM_HISTORY_BLOCK_INDEX' in os.environ or 'SLURM_ARRAY_TASK_ID' in os.environ):
                index = int(os.environ['SLURM_ARRAY_TASK_ID'] if 'SLURM_ARRAY_TASK_ID' in os.environ
                            else os.environ['NEURONSIM_HISTORY_BLOCK_INDEX'])
                registry = settings['source_registry']
                copy(repo/registry['path'], Path('source_registry.yaml'), registry['sha256'])
                blocks = yaml.safe_load((work/'source_registry.yaml').read_text())['blocks']
                assert [b['source_rng_seed'] for b in blocks] == registry['source_seeds'] == list(range(95001,95033))
                assert 1 <= index <= 31  # Seed95001 is reused from the accepted pilot.
                block = blocks[index]; seed = block['source_rng_seed']
                settings['source'] = block['coarse_references']['strong_cue']
                settings['source_rng_seed'] = seed
                settings['case_id'] = 'pfc_'+str(seed)+'_strong_cue'
                settings['sampling_seed_components'][1] = seed
                settings['resolved_source_registry_block_index'] = index
                analysis_config = work/'resolved_analysis_config.yaml'
                analysis_config.write_text(yaml.safe_dump(settings,sort_keys=False))
            origin = repo/settings['source']['archive']
            copy(origin/'delivery.json', Path('source_delivery.json'), settings['source']['delivery_sha256'])
            sd = json.loads((work/'source_delivery.json').read_text())
            if task == 'fine_history':
                assert sd['verified']
                case_prefix = settings['source']['case_path']
            else:
                assert sd['source_execution_complete'] and sd['exit_code'] == 0
                case_prefix = 'source_run'
            se = {row['path']:row for row in sd['files']}
            source_files = ('report.json','config.yaml','initial_voltage.npy') if task == 'suppression' else ('report.json','config.yaml','initial_voltage.npy','raw.npz')
            for name in source_files:
                key = case_prefix+'/'+name
                copy(origin/key, Path('source_inputs')/name, se[key]['sha256'])
            if task in ('witness','suppression'):
                assert digest(work/'source_inputs/config.yaml') == digest(work/'source_config.yaml')
            if settings['source'].get('report_sha256'):
                assert digest(work/'source_inputs/report.json') == settings['source']['report_sha256']
            for name in ('pfc_deterministic_replay.py','pfc_native_drive.py'):
                key = 'Libs/population_dynamics/'+name
                bound_module = work/'source_oracle.py' if task == 'suppression' and name == 'pfc_deterministic_replay.py' else work/key
                assert digest(bound_module) == se[key]['sha256']
            arguments = [sys.executable,'-B','-m','Libs.population_dynamics.'+stem,
                '--config',str(analysis_config),'--source',str(work/'source_inputs'),
                '--output',str(work/'validation')]
            if task in ('witness','suppression'):
                arguments.extend(['--reference',str(work/'reference'),'--witnesses',str(work/'first_hit_witnesses.npz')])
            elif task == 'fine_history':
                arguments.extend(['--paired-frame',str(work/'paired_coarse_frame.json')])
            if task == 'suppression': arguments.extend(['--witness-source',str(work/'witness_inputs')])
            phase('source_'+task, arguments)
    status = 'COMPLETE_'+status_name+'_'+mode.upper(); exit_code = 0
except Exception as exc:
    error = dict(type=type(exc).__name__,message=str(exc)); traceback.print_exc()
finally:
    suites = []
    if (work/'tests.xml').exists(): suites = [dict(node.attrib) for node in ET.parse(work/'tests.xml').getroot().iter('testsuite')]
    report = dict(status=status, exit_code=exit_code, error=error, job_id=job, node=socket.gethostname(),
        mode=mode, task=task, inputs=inputs, commands=commands, test_suites=suites, wall_seconds=time.perf_counter()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(), new_full_network_episodes=0,
        new_independent_source_blocks=0, population_history_uncertainty_established=False,
        confirmation_execution_released=False, production_episodes=0)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        archive.mkdir(parents=True,exist_ok=False); files = []
        for path in sorted(work.rglob('*')):
            if not path.is_file(): continue
            relative = path.relative_to(work)
            if relative.parts[0] in ('test_tmp','scratch','source_inputs','witness_inputs') or '__pycache__' in relative.parts: continue
            target = archive/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
            checksum = digest(path)
            assert digest(target) == checksum
            files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=checksum))
        (archive/'delivery.json').write_text(json.dumps(dict(status=status,verified=True,job_id=job,files=files),indent=2)+'\n')
        delivered = True; print(json.dumps(dict(status=status,archive=str(archive),exit_code=exit_code)),flush=True)
    finally:
        if delivered: shutil.rmtree(work)
        else: print('COPYBACK_FAILED_SCRATCH_PRESERVED:'+str(work),file=sys.stderr)
raise SystemExit(exit_code)
PY_WITNESS

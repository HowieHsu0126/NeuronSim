"""Pipeline tests with real small subprocesses and a mocked model server.

The subprocess fixture writes synthetic receipts; it imports no AppWorld and
generates no model output. It verifies ownership and ordering, not task success.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import yaml

from population_dynamics import appworld_benchmark_pipeline as pipeline


_CHILD_PROGRAM = r'''
import json
import os
from pathlib import Path
import sys
import time
role, specification_path, attempt_path = sys.argv[1:]
spec = json.loads(Path(specification_path).read_text())
attempt = Path(attempt_path)
if role == 'actor':
    attempt.mkdir()
    (attempt/'actor_pid.txt').write_text(str(os.getpid()))
    version = attempt/'experiments/outputs/development/tasks/traincase_1/version'
    version.mkdir(parents=True)
    (version/'code.txt').write_text('synthetic-native-code')
    (version/'data.txt').write_text('synthetic-database-version')
    logs = version.parent/'logs'
    logs.mkdir()
    (logs/'api_calls.jsonl').write_text('{"synthetic": true}\n')
    receipt = dict(spec['execution'], run_root=str(attempt))
    if spec.get('wrong_actor_identity'):
        receipt['task_id'] = 'traincase_2'
    if not spec.get('missing_receipt'):
        (attempt/'execution_receipt.json').write_text(json.dumps(receipt))
    # The receipt already exists while this actor is still alive. A scorer
    # started merely on file appearance would fail the process check below.
    time.sleep(0.04)
    sys.exit(spec.get('actor_exit', 0))
pid = int((attempt/'actor_pid.txt').read_text())
try:
    os.kill(pid, 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError('actor was still alive when the scorer started')
execution = json.loads((attempt/'execution_receipt.json').read_text())
scoreable = execution['status'] in {'completed', 'agent_failure'}
status = 'completed' if scoreable else 'not_run'
native_tgc = spec.get('native_tgc', False) if scoreable else None
outcome = {
    'schema': 'appworld_development_score_v1', 'scope': 'development_only', 'formal_eligible': False,
    **{key: execution[key] for key in ('split', 'task_id', 'seed', 'method', 'status')},
    'scoring_status': status, 'native_tgc': native_tgc, 'evaluator_called': scoreable,
}
if spec.get('evaluator_failure'):
    outcome.update(status='evaluator_failure', scoring_status='evaluator_failure', native_tgc=None)
if spec.get('wrong_score_identity'):
    outcome['seed'] = 42
(attempt/'development_score.json').write_text(json.dumps(outcome))
sys.exit(1 if spec.get('evaluator_failure') else 0)
'''


@pytest.fixture
def pipeline_fixture(tmp_path, monkeypatch):
    config = pipeline.load_benchmark_config(Path(__file__).resolve().parents[1] / 'configs/neural_population_appworld_benchmark.yaml')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    data = tmp_path / 'data_root'
    data.mkdir()
    native = {
        'status': 'PREPARED_RUNTIME_AND_SPLIT_METADATA', 'source_revision': config['source']['inspected_revision'],
        'requested_data_version': config['source']['candidate_data_version'],
        'packages': {'appworld': config['source']['candidate_package_version']},
        'hostname': socket.gethostname(), 'data_root': str(data),
        'native_python': sys.executable, 'source_checkout': str(tmp_path / 'source'),
    }
    manifest = {'schema': 'appworld_split_membership_v1', 'appworld_version': native['packages']['appworld'],
                'data_root': str(data), 'splits': {}}
    for split, prefix in (('train', 'traincase'), ('dev', 'devcase'), ('test_normal', 'normalcase'), ('test_challenge', 'challengecase')):
        manifest['splits'][split] = {'task_ids': [f'{prefix}_{variant}' for variant in (1,2,3)],
                                     'membership_sha256': 'a'*64, 'task_count': 3, 'scenario_count': 1}
    qualification = {
        'schema': 'appworld_model_runtime_qualification_v1', 'scope': 'neutral_model_transport_only',
        'status': 'PASS', 'tool_calls_validated': True, 'transport_probe_status': 'PASS',
        'seeds': [41,42,43], 'model_identity': deepcopy(config['model']),
        'config': {'model_qualification': deepcopy(config['model_qualification'])},
        'server_cleanup': {'server_parent_reaped': True, 'forced_kill': False,
                           'process_group_term_sent': True, 'returncode_before_termination': None},
        'runtime': {'packages': {'vllm': config['model']['serving_version'], 'torch': 'synthetic-torch'}},
    }
    model_runtime = {'status': 'MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED',
                     'packages': deepcopy(qualification['runtime']['packages'])}
    paths = {}
    for name, value in (('native_environment', native), ('split_manifest', manifest),
                        ('model_qualification', qualification), ('runtime_receipt', model_runtime)):
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(value))
    for name in ('source_metadata', 'snapshot_config'):
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text('{}')
    paths['model_directory'] = tmp_path / 'model'
    paths['model_directory'].mkdir()
    paths.update(config_path=config_path, output=tmp_path / 'run')
    state = {
        'paths': paths, 'config': config, 'native': native, 'manifest': manifest, 'qualification': qualification,
        'model_runtime': model_runtime,
        'server_events': [], 'server_alive': False, 'cleanup_failure': False, 'native_commands': [],
    }
    spec = {
        'execution': {'schema': 'appworld_development_execution_v1', 'scope': 'development_only', 'formal_eligible': False,
                      'split': 'train', 'task_id': 'traincase_1', 'seed': 41, 'method': 'official_react_code',
                      'terminal': True, 'status': 'completed', 'usage_complete': True,
                      'native_tgc': None, 'scoring_status': 'not_run', 'lm_call_count': 1,
                      'data_root': str(data), 'experiment_name': 'development', 'appworld_version': manifest['appworld_version']},
        'native_tgc': False,
    }
    child = tmp_path / 'synthetic_child.py'
    child.write_text(_CHILD_PROGRAM)
    specification = tmp_path / 'synthetic_spec.json'
    state.update(spec=spec, specification=specification)
    real_run_child = pipeline.run_child

    def execute_fixture(command, **kwargs):
        assert state['server_alive']
        assert kwargs['environment']['CUDA_VISIBLE_DEVICES'] == ''
        state['native_commands'].append(list(command))
        role = kwargs['name']
        assert command[0] == sys.executable
        assert command[1:3] == ['-m', 'population_dynamics.appworld_benchmark_development' if role == 'actor' else 'population_dynamics.appworld_benchmark_scoring']
        if role == 'actor':
            assert Path(command[command.index('--config')+1]).is_absolute()
        specification.write_text(json.dumps(spec))
        real_run_child([sys.executable, str(child), role, str(specification), str(paths['output']/'attempt')], **kwargs)

    @contextmanager
    def server(config, *, report, **kwargs):
        state['server_events'].append('start')
        state['server_alive'] = True
        report.update(server_pid=12345, stage='server_ready')
        try:
            yield 'http://127.0.0.1:1/v1'
        finally:
            state['server_alive'] = False
            state['server_events'].append('stop')
            report['server_cleanup'] = {'server_parent_reaped': True, 'forced_kill': state['cleanup_failure']}
            if state['cleanup_failure']:
                report.update(status='FAIL', cleanup_error='synthetic cleanup failure')

    monkeypatch.setattr(pipeline, 'owned_model_server', server)
    monkeypatch.setattr(pipeline, '_check_execution_location', lambda path: None)
    monkeypatch.setattr(pipeline, 'run_child', execute_fixture)
    return state


def run_fixture(state, **overrides):
    arguments = dict(state['paths'], agent='official_react_code', split='train', task_id='traincase_1', seed=41)
    arguments.update(overrides)
    return pipeline.run_development_pipeline(**arguments)


def test_development_selection_preserves_official_order_and_excludes_prompt_demos(pipeline_fixture):
    state = pipeline_fixture
    state['config']['development_execution']['api_predictor']['demo_task_ids'] = ['traincase_1']
    state['config']['development_execution']['methods']['official_full_code']['demo_task_ids'] = ['traincase_2']
    selection = pipeline.select_development_task(state['config'], state['manifest'], split='train', index=0)
    assert selection['task_id'] == 'traincase_3'
    assert selection['eligible_task_count'] == 1 and selection['native_membership_index'] == 2
    assert selection['excluded_demo_task_ids'] == ['traincase_1', 'traincase_2']
    assert selection['task_content_read'] is False and selection['scores_read'] is False
    assert pipeline.select_development_task(state['config'], state['manifest'], split='dev', index=1)['task_id'] == 'devcase_2'
    state['manifest']['splits']['dev']['task_ids'] = ['devcase_3', 'devcase_2', 'devcase_1']
    assert pipeline.select_development_task(state['config'], state['manifest'], split='dev', index=0)['task_id'] == 'devcase_3'


@pytest.mark.parametrize('split,index', [('test_normal',0), ('test_challenge',0), ('train',-1), ('train',True), ('dev',3)])
def test_development_selection_rejects_protected_or_invalid_requests(pipeline_fixture, split, index):
    state = pipeline_fixture
    with pytest.raises(ValueError):
        pipeline.select_development_task(state['config'], state['manifest'], split=split, index=index)


@pytest.mark.parametrize('native_tgc', [False, True])
def test_real_child_process_ordering_preserves_both_native_score_values(pipeline_fixture, native_tgc):
    state = pipeline_fixture
    state['spec']['native_tgc'] = native_tgc
    result = run_fixture(state)
    assert result['status'] == 'SCORED_DEVELOPMENT_ATTEMPT' and result['native_tgc'] is native_tgc
    assert state['server_events'] == ['start','stop'] and not state['server_alive']
    assert result['processes']['actor']['pid'] != result['processes']['scorer']['pid']
    assert all(row['cleanup']['parent_reaped'] and row['exit_code'] == 0 for row in result['processes'].values())
    assert result['neutral_probe_repeated'] is False and result['formal_eligible'] is False
    assert result['native_api_log_rows'] == 1
    assert result['native_versions'] == {'code': 'synthetic-native-code', 'data': 'synthetic-database-version'}
    assert json.loads((state['paths']['output']/'pipeline.json').read_text()) == result


def test_nonzero_actor_exit_with_a_valid_agent_failure_is_scored_without_retry(pipeline_fixture):
    state = pipeline_fixture
    state['spec'].update(actor_exit=7, native_tgc=True)
    state['spec']['execution']['status'] = 'agent_failure'
    result = run_fixture(state)
    assert result['status'] == 'SCORED_DEVELOPMENT_ATTEMPT' and result['native_tgc'] is True
    assert result['processes']['actor']['exit_code'] == 7 and result['execution_status'] == 'agent_failure'
    assert len(state['native_commands']) == 2


def test_function_calling_pipeline_passes_qualification_to_native_actor(pipeline_fixture):
    state = pipeline_fixture
    state['spec']['execution']['method'] = 'official_function_calling'
    result = run_fixture(state, agent='official_function_calling')
    assert result['status'] == 'SCORED_DEVELOPMENT_ATTEMPT'
    command = state['native_commands'][0]
    assert command[command.index('--model-qualification')+1] == str(state['paths']['model_qualification'])


@pytest.mark.parametrize('failure', ['missing_receipt','wrong_actor_identity'])
def test_unknown_or_wrong_actor_output_never_repeats_actor_or_starts_scorer(pipeline_fixture, failure):
    state = pipeline_fixture
    state['spec'][failure] = True
    result = run_fixture(state)
    assert result['status'] == 'FAILED' and result['native_tgc'] is None
    assert len(state['native_commands']) == 1 and 'scorer' not in result['processes']
    assert state['server_events'] == ['start','stop']


def test_unknown_native_execution_produces_an_explicit_unscored_outcome(pipeline_fixture):
    state = pipeline_fixture
    state['spec']['execution']['status'] = 'unknown'
    state['spec']['actor_exit'] = 1
    result = run_fixture(state)
    assert result['status'] == 'INCOMPLETE_DEVELOPMENT_ATTEMPT' and result['native_tgc'] is None
    assert result['outcome']['scoring_status'] == 'not_run'
    assert result['outcome']['evaluator_called'] is False and len(state['native_commands']) == 2


def test_evaluator_failure_remains_unscored_without_retry(pipeline_fixture):
    state = pipeline_fixture
    state['spec']['evaluator_failure'] = True
    result = run_fixture(state)
    assert result['status'] == 'INCOMPLETE_DEVELOPMENT_ATTEMPT' and result['native_tgc'] is None
    assert result['outcome']['status'] == 'evaluator_failure'
    assert result['processes']['scorer']['exit_code'] == 1 and len(state['native_commands']) == 2


def test_score_identity_mismatch_cannot_be_reported_as_success(pipeline_fixture):
    state = pipeline_fixture
    state['spec'].update(native_tgc=True, wrong_score_identity=True)
    result = run_fixture(state)
    assert result['status'] == 'FAILED' and result['native_tgc'] is None


def test_server_cleanup_failure_withholds_pipeline_success_but_keeps_valid_task_score(pipeline_fixture):
    state = pipeline_fixture
    state['cleanup_failure'] = True
    state['spec']['native_tgc'] = True
    result = run_fixture(state)
    assert result['status'] == 'FAILED_RUNTIME_CLEANUP' and result['native_tgc'] is True
    assert result['outcome']['scoring_status'] == 'completed'


@pytest.mark.parametrize('defect', ['qualification','model_revision','serving_settings','cleanup','native_revision','host','seeds',
                                  'different_runtime','missing_qualified_packages','unprepared_runtime'])
def test_unadmitted_inputs_reject_before_server_start(pipeline_fixture, defect):
    state = pipeline_fixture
    qualification, native, config = deepcopy(state['qualification']), deepcopy(state['native']), deepcopy(state['config'])
    if defect == 'qualification': qualification['status'] = 'FAIL'
    elif defect == 'model_revision': qualification['model_identity']['revision'] = 'different'
    elif defect == 'serving_settings': qualification['config']['model_qualification']['max_model_len'] += 1
    elif defect == 'cleanup': qualification['server_cleanup']['forced_kill'] = True
    elif defect == 'native_revision': native['source_revision'] = 'different'
    elif defect == 'host': native['hostname'] = 'different-node'
    elif defect == 'seeds': config['seeds'] = [41,42,44]
    elif defect == 'missing_qualified_packages': qualification.pop('runtime')
    else:
        model_runtime = deepcopy(state['model_runtime'])
        if defect == 'different_runtime': model_runtime['packages']['torch'] = 'different-torch'
        else: model_runtime['status'] = 'INCOMPLETE'
        state['paths']['runtime_receipt'].write_text(json.dumps(model_runtime))
    state['paths']['model_qualification'].write_text(json.dumps(qualification))
    state['paths']['native_environment'].write_text(json.dumps(native))
    state['paths']['config_path'].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        run_fixture(state)
    assert state['server_events'] == [] and not state['paths']['output'].exists()


def test_formal_tasks_reject_even_if_configuration_release_flag_is_true(pipeline_fixture):
    state = pipeline_fixture
    state['config']['formal']['released'] = True
    state['paths']['config_path'].write_text(yaml.safe_dump(state['config']))
    with pytest.raises(ValueError, match='development'):
        run_fixture(state, split='test_normal', task_id='normalcase_1')
    assert state['server_events'] == []


def test_existing_output_is_not_overwritten(pipeline_fixture):
    state = pipeline_fixture
    output = state['paths']['output']
    output.mkdir()
    (output/'retained.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        run_fixture(state)
    assert (output/'retained.txt').read_text() == 'keep' and state['server_events'] == []


def test_actual_pipeline_cli_accepts_relative_input_paths_and_a_valid_negative(pipeline_fixture, monkeypatch, capsys):
    state = pipeline_fixture
    root = state['paths']['config_path'].parent
    monkeypatch.chdir(root)
    arguments = []
    for key, path in state['paths'].items():
        name = 'config' if key == 'config_path' else key.replace('_','-')
        arguments += ['--'+name, str(path.relative_to(root))]
    arguments += ['--agent','official_react_code','--split','train','--task-id','traincase_1','--seed','41']
    assert pipeline.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['native_tgc'] is False and result['status'] == 'SCORED_DEVELOPMENT_ATTEMPT'


def test_collaboration_command_routes_to_its_own_cli_with_explicit_controls(tmp_path):
    command = pipeline.actor_command(Path(sys.executable), config_path=tmp_path/'config', split_manifest=tmp_path/'splits',
        source_root=tmp_path/'source', attempt_root=tmp_path/'attempt', endpoint='http://127.0.0.1:1/v1',
        agent='collaboration_fixed', split='train', task_id='synthetic_1', seed=43,
        verification_budget=2, planner_calls=0, max_model_calls=50)
    assert command[2] == 'population_dynamics.appworld_benchmark_collaboration_development'
    assert command[command.index('--policy')+1] == 'fixed'
    assert command[command.index('--planner-calls')+1] == '0'
    assert command[command.index('--seed')+1] == '43'


def test_interrupted_real_child_is_terminated_and_reaped(tmp_path, monkeypatch):
    original = subprocess.Popen
    children = []
    class InterruptedWait(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_wait = True
            children.append(self)
        def wait(self, *args, **kwargs):
            if self.first_wait:
                self.first_wait = False
                raise KeyboardInterrupt('synthetic parent interruption')
            return super().wait(*args, **kwargs)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', InterruptedWait)
    record = {}
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_child([sys.executable,'-c','import time; time.sleep(60)'], output=tmp_path,
                           name='owned', environment={}, record=record, progress=lambda: None)
    assert len(children) == 1 and children[0].poll() is not None
    assert record['cleanup']['parent_reaped'] is True and record['wall_seconds'] >= 0

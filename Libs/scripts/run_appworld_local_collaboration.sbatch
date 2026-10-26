#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldLocal
#SBATCH --output=/tmp/neuronsim-appworld-local-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-local-%j.err
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH hetjob
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=2G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp

# Submit only after the exported pinned readiness report and native transport589479
# have passed verification and the requested nodes have passed fresh admission.
# This script does not submit/requeue jobs or assert that those jobs have passed.
# Required exported inputs: APPWORLD_GPU_PYTHON, APPWORLD_MODEL_METADATA,
# APPWORLD_READINESS_REPORT. Required positional argument: planner quota 0, 1, 2.
# The executor quota is 2 minus that value; the split stays hidden until proposal.
# A separately registered repair uses APPWORLD_OBSERVATION_HISTORY_REPAIR;
# APPWORLD_VERIFICATION_REPLY_PROTOCOL selects the separately registered coordination
# diagnostic; APPWORLD_SCRATCH_ROOT may select /data/neuronsim on both allocated nodes.
# APPWORLD_GPU_NODE selects gpu03 (default) or cpu03; CPU control stays on cpu04.
# The submitted heterogeneous allocation must use the same selected GPU node.
# Native and GPU Python CLI exit2 means completed development execution. Raw
# codes are archived; batch success requires exit2 and successful verified copy-out.
set -uo pipefail

REPO=/workspace/NeuronSim
CPU_PYTHON=/tmp/neuronsim/neuronsim-appworld-runtime-575402/venv/bin/python
CPU_BENCHMARK=/tmp/neuronsim/neuronsim-appworld-runtime-575402/benchmark
MODEL_DIRECTORY=/data/neuronsim/neuronsim_models/qwen3_coder_30b_a3b_b2cff646
EXECUTION_LAUNCHER_SOURCE=${BASH_SOURCE[0]}
# Slurm can normalize its batch copy. Keep the validated source bytes separate.
LAUNCHER_SOURCE=${APPWORLD_LAUNCHER_SOURCE:-$EXECUTION_LAUNCHER_SOURCE}
# The nodes use the same scratch name, so the two roles must remain separate.
export APPWORLD_GPU_NODE=${APPWORLD_GPU_NODE:-gpu03}
case "$APPWORLD_GPU_NODE" in
    gpu03|cpu03) ;;
    cpu04)
        printf '%s\n' 'APPWORLD_GPU_NODE_COLLIDES_WITH_CPU_CONTROLLER: cpu04' >&2
        exit 1
        ;;
    *)
        printf '%s\n' 'APPWORLD_GPU_NODE_UNSUPPORTED: select gpu03 or cpu03' >&2
        exit 1
        ;;
esac

configure_local_environment() {
    local work=$1
    cd "$work" || return 1
    export PYTHONPATH="$work"
    if [ -d "$work/decoder_wheels" ]; then
        export PYTHONPATH="$PYTHONPATH:$work/decoder_wheels/lm_format_enforcer-0.11.3-py3-none-any.whl:$work/decoder_wheels/interegular-0.3.3-py37-none-any.whl"
    fi
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export TMPDIR="$work"
    export APPWORLD_CACHE="$work/.cache/appworld"
    export XDG_CACHE_HOME="$work/.cache"
    export HF_HOME="$work/.cache/huggingface"
    export TORCH_HOME="$work/.cache/torch"
    export CUDA_CACHE_PATH="$work/.cache/cuda"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
}

helper() {
    local interpreter=$1
    shift
    run_and_wait "$interpreter" - "$@" <<'PY_HELPER'
from pathlib import Path
import hashlib
import json
import os
import shutil
import socket
import sys

mode, work_text, archive_text, role = sys.argv[1:5]
work, archive = Path(work_text), Path(archive_text)
job_id = os.environ.get('APPWORLD_HET_LEADER_JOB_ID', '')
if not job_id.isdigit():
    raise RuntimeError('SLURM_HETEROGENEOUS_JOB_ID_REQUIRED')
repo = Path('/workspace/NeuronSim')
revision = 'b2cff646eb4bb1d68355c01b18ae02e7cf42d120'
repo_id = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
identity = repo_id + '@' + revision
expected_node = {'gpu_host': os.environ['APPWORLD_GPU_NODE'], 'cpu_controller': 'cpu04'}[role]
if socket.gethostname().split('.')[0] != expected_node:
    raise RuntimeError('APPWORLD_COMPONENT_NODE_MISMATCH')
scratch_root = Path(os.environ.get('APPWORLD_SCRATCH_ROOT', '/tmp/neuronsim'))
if scratch_root not in (Path('/tmp/neuronsim'), Path('/data/neuronsim')):
    raise RuntimeError('APPWORLD_UNSUPPORTED_LOCAL_SCRATCH_ROOT')
if work != scratch_root / ('neuronsim-appworld-local-' + job_id):
    raise RuntimeError('APPWORLD_LOCAL_WORK_PATH_MISMATCH')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def copy_verified(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    value = digest(source)
    if digest(target) != value or (expected is not None and value != expected):
        raise RuntimeError('APPWORLD_STAGED_COPY_HASH_MISMATCH:' + str(target))
    return {'path': str(target.relative_to(target_root)), 'bytes': target.stat().st_size, 'sha256': value}


def health():
    blocked = 0
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            blocked += path.read_text().rsplit(')', 1)[1].split()[0] == 'D'
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in {'MemTotal', 'MemAvailable'}:
            memory[name + '_bytes'] = int(value.split()[0]) * 1024
    space = os.statvfs(work)
    root = os.statvfs('/')
    root_free = root.f_bavail * root.f_frsize
    root_used_fraction = 1 - root.f_bfree / root.f_blocks
    write_json(work / 'resource.json', {
        'job_id': job_id, 'role': role, 'hostname': socket.gethostname(),
        'd_state': blocked, 'memory': memory,
        'scratch_available_bytes': space.f_bavail * space.f_frsize,
        'root_available_bytes': root_free, 'root_used_fraction': root_used_fraction,
    })
    if root_free < 20 * 1024**3 or root_used_fraction >= 0.95:
        raise RuntimeError('COMPUTE_ROOT_STORAGE_REQUIRES_RECOVERY')
    if blocked >= 15:
        raise RuntimeError('COMPUTE_NODE_HEALTH_REQUIRES_RECHECK')


def archive_node(step_code):
    destination = archive / role
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for directory, dirs, names in os.walk(work, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in {'.cache', '__pycache__'})
        directory = Path(directory)
        for name in sorted([*dirs, *names]):
            source = directory / name
            relative = source.relative_to(work)
            target = destination / relative
            if source.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.readlink(source))
                if os.readlink(target) != os.readlink(source):
                    raise RuntimeError('APPWORLD_ARCHIVE_SYMLINK_MISMATCH')
                continue
            if not source.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            value = digest(source)
            if digest(target) != value:
                raise RuntimeError('APPWORLD_ARCHIVE_FILE_HASH_MISMATCH')
            files.append({'path': str(relative), 'bytes': source.stat().st_size, 'sha256': value})
    receipt = {'job_id': job_id, 'role': role, 'gpu_host_step_exit_code': step_code,
               'files': files, 'verified': True, 'scratch_removed_after_copy': False}
    write_json(destination / 'copy_out.json', receipt)
    os.chdir(work.parent)
    shutil.rmtree(work)
    receipt['scratch_removed_after_copy'] = True
    (destination / 'copy_out.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'archived': str(destination), 'files': len(files), 'gpu_host_step_exit_code': step_code}), file=sys.stderr)


if mode == 'stage_gpu':
    import yaml
    if work.exists() or archive.exists():
        raise RuntimeError('APPWORLD_EXISTING_WORK_OR_ARCHIVE_REQUIRES_INSPECTION')
    archive.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    try:
        health()
        planner_quota = int(sys.argv[5])
        if planner_quota not in {0, 1, 2}:
            raise ValueError('APPWORLD_PLANNER_QUOTA_INVALID')
        sources = [
            'Libs/__init__.py', 'Libs/population_dynamics/__init__.py',
            'Libs/population_dynamics/provenance.py',
            'Libs/population_dynamics/appworld_reference.py',
            'Libs/population_dynamics/appworld_collaboration.py',
            'Libs/population_dynamics/appworld_local_actor.py',
            'Libs/population_dynamics/appworld_action_decoder.py',
            'Libs/population_dynamics/appworld_local_host.py',
            'Libs/population_dynamics/appworld_recorded_prefix.py',
            'Libs/scripts/run_appworld_local_collaboration.py',
            'Libs/configs/neural_population_appworld_local_collaboration.yaml',
            'Libs/configs/neural_population_appworld_tool_actor.yaml',
        ]
        target_root = work
        for name in sources:
            copy_verified(repo / name, work / name)
        extra = {
            'api_docs.json': repo / 'Output/neural_population_dynamics/agentic_extension/appworld_two_agent_development/api_docs.json',
            'model_metadata.json': Path(os.environ['APPWORLD_MODEL_METADATA']),
            'readiness_report.json': Path(os.environ['APPWORLD_READINESS_REPORT']),
            'launcher.sbatch': Path(sys.argv[6]),
            'execution_launcher.sbatch': Path(sys.argv[7]),
        }
        for name, source in extra.items():
            if not source.is_absolute():
                raise ValueError('APPWORLD_ABSOLUTE_INPUT_PATH_REQUIRED:' + name)
            copy_verified(source, work / name)
            sources.append(name)
        write_json(work / 'launcher_source_binding.json', {
            'validated_source_path': str(extra['launcher.sbatch']),
            'validated_source_sha256': digest(work / 'launcher.sbatch'),
            'execution_wrapper_path': str(extra['execution_launcher.sbatch']),
            'execution_wrapper_sha256': digest(work / 'execution_launcher.sbatch'),
            'source_kind': 'explicit_validated_source' if os.environ.get('APPWORLD_LAUNCHER_SOURCE') else 'execution_source',
            'staged_helper_source': 'launcher.sbatch',
        })
        sources.append('launcher_source_binding.json')
        config = yaml.safe_load((work / 'Libs/configs/neural_population_appworld_local_collaboration.yaml').read_text())
        actor = yaml.safe_load((work / 'Libs/configs/neural_population_appworld_tool_actor.yaml').read_text())
        if (config['native']['task_id'] != '07b42fd_1' or config['native']['dataset'] != 'train'
                or config['provider'] != 'local_transformers' or config['model_revision'] != identity
                or config['verification_budget'] != 2
                or actor['model']['repo_id'] != repo_id or actor['model']['revision'] != revision
                or actor['model']['cuda_devices'] != 2):
            raise ValueError('APPWORLD_DEVELOPMENT_TASK_OR_MODEL_CONTRACT_MISMATCH')
        report = json.loads((work / 'readiness_report.json').read_text())
        metadata = json.loads((work / 'model_metadata.json').read_text())
        verification = report.get('snapshot_verification', {})
        if (report.get('status') != 'LOCAL_MODEL_INFERENCE_READY'
                or type(report.get('exit_code')) is not int or report['exit_code'] != 0
                or report.get('model_revision') != identity
                or verification.get('model_revision') != identity
                or verification.get('all_published_file_identities_verified') is not True
                or not verification.get('files')
                or report.get('source_metadata_sha256') != digest(work / 'model_metadata.json')
                or metadata.get('id') != repo_id or metadata.get('sha') != revision):
            raise ValueError('APPWORLD_PINNED_READINESS_NOT_VERIFIED')
        if not Path('/data/neuronsim/neuronsim_models/qwen3_coder_30b_a3b_b2cff646').is_dir():
            raise ValueError('APPWORLD_PINNED_MODEL_DIRECTORY_MISSING')
        (work / 'allocation.yaml').write_text(yaml.safe_dump({'planner': planner_quota, 'executor': 2 - planner_quota}))
        write_json(work / 'controller_command.json', [
            'srun', '--het-group=1', '--cpu-bind=cores', '--ntasks=1', '--cpus-per-task=1',
            '--mem=2G', '--gres=none', 'bash', str(work / 'launcher.sbatch'), '__native', str(work),
        ])
        sources.extend(['allocation.yaml', 'controller_command.json'])
        prefix_path = os.environ.get('APPWORLD_RECORDED_PREFIX_PROTOCOL')
        repair_path = os.environ.get('APPWORLD_OBSERVATION_HISTORY_REPAIR')
        reply_path = os.environ.get('APPWORLD_VERIFICATION_REPLY_PROTOCOL')
        decoder_path = os.environ.get('APPWORLD_PHASE_ACTION_DECODER_PROTOCOL')
        bounds_path = os.environ.get('APPWORLD_ACTION_DECODER_BOUNDS')
        if bounds_path and not decoder_path:
            raise ValueError('APPWORLD_BOUNDS_REQUIRE_DECODER_PROTOCOL')
        if sum(bool(path) for path in (repair_path, reply_path, decoder_path)) > 1:
            raise ValueError('APPWORLD_DISTINCT_DEVELOPMENT_PROTOCOL_REQUIRED')
        if (repair_path or reply_path or decoder_path) and not prefix_path:
            raise ValueError('APPWORLD_DEVELOPMENT_REQUIRES_VERIFIED_PREFIX')
        if prefix_path:
            prefix_path = Path(prefix_path)
            validation_path = Path(os.environ['APPWORLD_PREFIX_VALIDATION_REPORT'])
            if not prefix_path.is_absolute() or not validation_path.is_absolute():
                raise ValueError('APPWORLD_ABSOLUTE_PREFIX_INPUT_REQUIRED')
            prefix_settings = yaml.safe_load(prefix_path.read_text())
            validation = json.loads(validation_path.read_text())
            arm = os.environ.get('APPWORLD_DIAGNOSTIC_ARM')
            development = None
            bounded = bool(bounds_path)
            development_path = Path(decoder_path or reply_path or repair_path) if (decoder_path or reply_path or repair_path) else None
            if development_path:
                if not development_path.is_absolute():
                    raise ValueError('APPWORLD_ABSOLUTE_DEVELOPMENT_SETTINGS_REQUIRED')
                development = yaml.safe_load(development_path.read_text())
                expected = {
                    'schema': ('appworld_bounded_message_decoder_development_v1' if bounded else
                               'appworld_phase_action_decoder_development_v1' if decoder_path else
                               'appworld_verification_message_reply_development_v1' if reply_path
                               else 'appworld_observation_history_development_v1'),
                    'task_id': config['native']['task_id'], 'dataset': 'train',
                    'model_revision': identity,
                    ('historical_reference_job' if (decoder_path or reply_path) else 'historical_baseline_job'): (597334 if bounded else 596889 if decoder_path else 592355 if reply_path else 591164),
                    'prefix_protocol_sha256': digest(prefix_path),
                    'prefix_generation_attempts': 10, 'maximum_new_attempts': 20,
                    'maximum_total_attempts': 30, 'maximum_new_executions': 1,
                    'allocation': {'planner': planner_quota, 'executor': 2-planner_quota},
                    'lossless_observation_history': True, 'evidence_table_after_prefix': True,
                    'max_context_tokens': actor['generation']['max_context_tokens'],
                    'max_new_tokens': actor['generation']['max_new_tokens'],
                }
                if reply_path or decoder_path:
                    expected.update(verification_message_replies=True, reply_generation_attempts_per_delivered_message=1)
                if decoder_path:
                    expected.update(phase_action_decoder=True)
                if bounded:
                    expected.update(action_decoder_bounds={'message_text_max_length':256,'max_consecutive_whitespaces':4})
                    bounds_source = Path(bounds_path)
                    if not bounds_source.is_absolute():
                        raise ValueError('APPWORLD_ABSOLUTE_BOUNDS_INPUT_REQUIRED')
                    bounds = yaml.safe_load(bounds_source.read_text())
                    if bounds != expected['action_decoder_bounds']:
                        raise ValueError('APPWORLD_BOUNDS_PROTOCOL_MISMATCH')
                if (arm is not None or not isinstance(development, dict)
                        or any(development.get(key) != value for key, value in expected.items())
                        or validation.get('lossless_observation_history_requested') is not True
                        or ((reply_path or decoder_path) and validation.get('verification_message_replies_requested') is not True)
                        or (decoder_path and (validation.get('phase_action_decoder_requested') is not True
                            or validation.get('real_tokenizer_decoder_validation', {}).get('status') != 'PASS'))):
                    raise ValueError('APPWORLD_DEVELOPMENT_SCOPE_OR_VALIDATION_MISMATCH')
            if (prefix_settings.get('schema') != 'appworld_recorded_prefix_diagnostic_v1'
                    or prefix_settings.get('source_case_job') != 590688
                    or prefix_settings.get('prefix_attempts') != 10 or prefix_settings.get('remaining_attempts') != 20
                    or prefix_settings.get('maximum_task_prefix_pairs') != 1
                    or (development is None and arm not in prefix_settings.get('arms', []))
                    or prefix_settings.get('allocation') != {'planner': planner_quota, 'executor': 2-planner_quota}
                    or validation.get('status') != ('RECORDED_PREFIX_REUSED_CURRENT_HOST_VALIDATED' if bounded else 'RECORDED_PREFIX_NATIVE_VALIDATION_COMPLETE')
                    or validation.get('exit_code') != 0 or validation.get('live_model_generations') != 0
                    or validation.get('host_prefix', {}).get('status') != 'VERIFIED'
                    or validation.get('native_prefix', {}).get('status') != 'PASS'):
                raise ValueError('APPWORLD_VERIFIED_NATIVE_PREFIX_REQUIRED')
            if bounded and (validation.get('new_native_executions') != 0
                    or validation.get('new_model_generations') != 0
                    or validation.get('current_prefix_replay',{}).get('status') != 'PASS_RETAINED_NATIVE_TRANSPORT_AND_CURRENT_PRIVATE_PREFIX'
                    or validation.get('bounded_text_validation',{}).get('status') != 'PASS_TARGETED_BOUNDED_TEXT_GENERATION'):
                raise ValueError('APPWORLD_CURRENT_BOUNDED_HOST_VALIDATION_REQUIRED')
            verified_inputs = {entry['path']: entry for entry in validation['inputs']}
            for name in ['Libs/population_dynamics/appworld_collaboration.py',
                         'Libs/population_dynamics/appworld_local_actor.py',
                         'Libs/population_dynamics/appworld_local_host.py',
                         'Libs/population_dynamics/appworld_recorded_prefix.py',
                         'Libs/scripts/run_appworld_local_collaboration.py',
                         'Libs/configs/neural_population_appworld_tool_actor.yaml',
                         'Libs/configs/neural_population_appworld_local_collaboration.yaml',
                         'api_docs.json', 'allocation.yaml', 'model_metadata.json']:
                if digest(work / name) != verified_inputs[name]['sha256']:
                    raise ValueError('APPWORLD_PREFIX_VALIDATED_CODE_CHANGED:' + name)
            if digest(prefix_path) != verified_inputs['prefix/protocol.yaml']['sha256']:
                raise ValueError('APPWORLD_PREFIX_PROTOCOL_CHANGED')
            copy_verified(prefix_path, work / 'prefix/protocol.yaml')
            sources.append('prefix/protocol.yaml')
            for binding in prefix_settings['inputs'].values():
                relative = Path(binding['path'])
                if relative.is_absolute() or len(relative.parts) != 1:
                    raise ValueError('APPWORLD_PREFIX_INPUT_NAME_INVALID')
                name = 'prefix/' + str(relative)
                copy_verified(prefix_path.parent / relative, work / name)
                if digest(work / name) != binding['sha256']:
                    raise ValueError('APPWORLD_PREFIX_INPUT_CHANGED:' + name)
                sources.append(name)
            copy_verified(validation_path, work / 'prefix_validation_report.json')
            sources.append('prefix_validation_report.json')
            if development is not None:
                name = 'phase_action_decoder_protocol.yaml' if decoder_path else 'verification_reply_protocol.yaml' if reply_path else 'observation_history_repair.yaml'
                if (reply_path or decoder_path) and digest(development_path) != verified_inputs.get(name, {}).get('sha256'):
                    raise ValueError('APPWORLD_DEVELOPMENT_PROTOCOL_CHANGED')
                copy_verified(development_path, work / name)
                sources.append(name)
                if bounded:
                    copy_verified(bounds_source, work / 'action_decoder_bounds.yaml')
                    if digest(work / 'action_decoder_bounds.yaml') != verified_inputs.get('action_decoder_bounds.yaml',{}).get('sha256'):
                        raise ValueError('APPWORLD_BOUNDS_VALIDATED_CONFIGURATION_CHANGED')
                    sources.append('action_decoder_bounds.yaml')
                if decoder_path:
                    for code_name in ['Libs/population_dynamics/appworld_action_decoder.py', 'launcher.sbatch']:
                        if digest(work / code_name) != verified_inputs.get(code_name, {}).get('sha256'):
                            raise ValueError('APPWORLD_DECODER_VALIDATED_CODE_CHANGED:' + code_name)
                    dependencies = development['decoder']
                    origin = repo / dependencies['dependency_archive']
                    if digest(origin / 'delivery.json') != dependencies['dependency_delivery_sha256']:
                        raise ValueError('APPWORLD_DECODER_DEPENDENCY_DELIVERY_CHANGED')
                    if dependencies.get('implementation') != 'lm-format-enforcer' or dependencies.get('version') != '0.11.3':
                        raise ValueError('APPWORLD_DECODER_IMPLEMENTATION_CHANGED')
                    for wheel, expected_hash in dependencies['wheels'].items():
                        name = 'decoder_wheels/' + wheel
                        copy_verified(origin / wheel, work / name, expected_hash)
                        if digest(work / name) != verified_inputs.get(name, {}).get('sha256'):
                            raise ValueError('APPWORLD_DECODER_VALIDATED_WHEEL_CHANGED')
                        sources.append(name)
            else:
                (work / 'diagnostic_arm.txt').write_text(arm + '\n')
                sources.append('diagnostic_arm.txt')
        snapshot = archive / 'inputs'
        snapshot.mkdir(exist_ok=False)
        target_root = snapshot
        files = [copy_verified(work / name, snapshot / name) for name in sources]
        receipt = {'job_id': job_id, 'scope': 'same_frozen_inputs_for_gpu_host_and_cpu_controller',
                   'files': files, 'verified': True, 'scratch_removed_after_copy': False}
        write_json(snapshot / 'copy_out.json', receipt)
        shutil.copy2(snapshot / 'copy_out.json', work / 'frozen_inputs_copy_out.json')
        print(json.dumps({'staged': role, 'work': str(work), 'snapshot': str(snapshot), 'readiness_verified': True}), file=sys.stderr)
    except Exception:
        try:
            archive_node(None)
        except Exception as archive_error:
            print(json.dumps({'partial_stage_archive_error': type(archive_error).__name__, 'scratch_retained': str(work)}), file=sys.stderr)
        raise
elif mode == 'stage_cpu':
    if work.exists():
        raise RuntimeError('APPWORLD_EXISTING_CPU_WORK_REQUIRES_INSPECTION')
    work.mkdir(parents=True, exist_ok=False)
    try:
        health()
        runtime = Path('/tmp/neuronsim/neuronsim-appworld-runtime-575402')
        if not os.access(runtime / 'venv/bin/python', os.X_OK) or not (runtime / 'benchmark').is_dir():
            raise RuntimeError('APPWORLD_NATIVE_RUNTIME_MISSING')
        snapshot = archive / 'inputs'
        receipt = json.loads((snapshot / 'copy_out.json').read_text())
        if receipt.get('verified') is not True or receipt.get('job_id') != job_id:
            raise RuntimeError('APPWORLD_FROZEN_INPUT_RECEIPT_INVALID')
        target_root = work
        for row in receipt['files']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('APPWORLD_FROZEN_INPUT_PATH_INVALID')
            copy_verified(snapshot / relative, work / relative, row['sha256'])
        shutil.copy2(snapshot / 'copy_out.json', work / 'frozen_inputs_copy_out.json')
        print(json.dumps({'staged': role, 'work': str(work), 'same_frozen_input_hashes': True}), file=sys.stderr)
    except Exception:
        try:
            archive_node(None)
        except Exception as archive_error:
            print(json.dumps({'partial_stage_archive_error': type(archive_error).__name__, 'scratch_retained': str(work)}), file=sys.stderr)
        raise
elif mode == 'preserve_failure':
    # Capture a bounded snapshot. Do not recurse into native data or model caches.
    component, code, phase, signal = sys.argv[5:9]
    destination = archive / 'failure' / component
    destination.mkdir(parents=True, exist_ok=False)
    limit = 1024 * 1024
    metadata = [
        'process_exit.json', 'launcher_result.json', 'resource.json',
        'controller_command.json', 'launcher_source_binding.json', 'frozen_inputs_copy_out.json',
        'host/report.json', 'host/generation_attempts.json', 'host/manifest.json',
        'controller/report.json',
    ]
    logs = ['host/controller_stderr.log', 'host_stdout.log', 'host_stderr.log', 'native_stderr.log']
    captured, missing, errors = [], [], []
    sources = [(work / name, name, name in logs) for name in metadata + logs]
    if component == 'launcher':
        for suffix in ('out', 'err'):
            sources.append((Path('/tmp') / ('neuronsim-appworld-local-' + job_id + '.' + suffix),
                            'batch.' + suffix, True))
    for source, name, tail in sources:
        if not source.is_file() or source.is_symlink():
            missing.append(name)
            continue
        try:
            size = source.stat().st_size
            partial = size > limit
            target = destination / (name + '.partial' if partial else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as stream:
                if tail and partial:
                    stream.seek(size - limit)
                payload = stream.read(limit)
            with target.open('xb') as stream:
                stream.write(payload)
            captured.append({'source': name, 'path': str(target.relative_to(destination)),
                             'source_bytes_at_capture': size, 'captured_bytes': len(payload),
                             'truncated': partial, 'capture': 'tail' if tail and partial else 'prefix'})
        except OSError as error:
            errors.append({'source': name, 'error': type(error).__name__, 'detail': str(error)})
    write_json(destination / 'interruption.json', {
        'job_id': job_id, 'role': role, 'component': component,
        'shell_exit_code': int(code), 'phase': phase, 'signal': signal or None,
        'scratch_path': str(work), 'scratch_exists_at_capture': work.is_dir(),
        'scratch_removed': False, 'snapshot': 'bounded_existing_files_at_shell_exit',
        'per_file_capture_limit_bytes': limit, 'files': captured,
        'missing_files': missing, 'copy_errors': errors,
    })
    print(json.dumps({'failure_snapshot': str(destination), 'shell_exit_code': int(code),
                      'scratch_retained': str(work), 'copy_errors': len(errors)}), file=sys.stderr)
elif mode == 'record_exit':
    write_json(work / 'process_exit.json', {'job_id': job_id, 'role': role, 'python_cli_exit_code': int(sys.argv[5])})
elif mode == 'record_launcher':
    write_json(work / 'launcher_result.json', {
        'job_id': job_id, 'cpu_staging_step_exit_code': int(sys.argv[5]),
        'gpu_host_step_exit_code': int(sys.argv[6]), 'cpu_archive_step_exit_code': int(sys.argv[7]),
        'transport_probe_pre_submission_verification': 'owned_by_submitter_not_reasserted_by_launcher',
        'scientific_control_transfer_qualified': False, 'formal_model_comparison_qualified': False,
        'complete_host_actor_native_cost_qualified': False,
    })
elif mode == 'archive':
    archive_node(int(sys.argv[5]))
else:
    raise ValueError('APPWORLD_LAUNCHER_HELPER_MODE_INVALID')
PY_HELPER
}


# Arm before each mode can create or consume scratch. A component exit code of 2
# is accepted only after its real process receipt has been written.
install_failure_handler() {
    failure_work=$1
    failure_archive=$2
    failure_role=$3
    failure_component=$4
    failure_interpreter=$5
    failure_phase=entry
    failure_signal=
    failure_child_pid=
    failure_armed=1
    trap 'preserve_failure "$?"' EXIT
    trap 'interrupt_component TERM 143' TERM
    trap 'interrupt_component INT 130' INT
}

preserve_failure() {
    local result=$1
    trap - EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    if [ "$failure_armed" -eq 1 ] && [ -d "$failure_work" ]; then
        helper "$failure_interpreter" preserve_failure "$failure_work" "$failure_archive" \
            "$failure_role" "$failure_component" "$result" "$failure_phase" "$failure_signal" >&2
        local snapshot_code=$?
        if [ "$snapshot_code" -ne 0 ]; then
            printf 'APPWORLD_FAILURE_SNAPSHOT_FAILED exit=%s scratch=%s\n' "$snapshot_code" "$failure_work" >&2
        fi
    fi
    exit "$result"
}

interrupt_component() {
    failure_signal=$1
    # Forward the received signal to the existing local child. Do not start a
    # Slurm step or issue a scheduler mutation from this handler.
    if [ -n "$failure_child_pid" ]; then
        kill -s "$failure_signal" "$failure_child_pid" 2>/dev/null || :
        local check
        for ((check=0; check<20; check++)); do
            if ! kill -0 "$failure_child_pid" 2>/dev/null; then break; fi
            sleep 0.1
        done
        if kill -0 "$failure_child_pid" 2>/dev/null; then
            kill -KILL "$failure_child_pid" 2>/dev/null || :
        fi
        wait "$failure_child_pid" 2>/dev/null || :
        failure_child_pid=
    fi
    exit "$2"
}

run_and_wait() {
    # Bash background commands otherwise inherit ignored SIGINT. Restore the
    # child signals and retain the native controller's input stream.
    (trap - INT TERM; exec "$@") <&0 &
    failure_child_pid=$!
    wait "$failure_child_pid"
    local result=$?
    failure_child_pid=
    return "$result"
}

run_archive_path() {
    printf '%s/Output/neural_population_dynamics/agentic_extension/appworld_local_collaboration/run_%s\n' \
        "$REPO" "$APPWORLD_HET_LEADER_JOB_ID"
}

run_native() {
    local work=$1
    configure_local_environment "$work" || return 1
    export CUDA_VISIBLE_DEVICES=
    # Nothing except controller NDJSON may go to this function's stdout.
    local -a prefix_args=()
    if [ -f "$work/prefix/protocol.yaml" ]; then
        prefix_args+=(--prefix-checkpoint "$work/prefix/native_checkpoint.json")
    fi
    failure_phase=native_process
    run_and_wait "$CPU_PYTHON" -u -m Libs.population_dynamics.appworld_collaboration \
        --config "$work/Libs/configs/neural_population_appworld_local_collaboration.yaml" \
        --api-schemas "$work/api_docs.json" --data-root "$CPU_BENCHMARK" \
        "${prefix_args[@]}" --output "$work/controller" 2>"$work/native_stderr.log"
    local code=$?
    failure_phase=process_receipt
    helper "$CPU_PYTHON" record_exit "$work" "$work" cpu_controller "$code"
    local receipt_code=$?
    if [ "$receipt_code" -ne 0 ]; then return 1; fi
    if [ "$code" -eq 2 ]; then failure_armed=0; fi
    return "$code"
}

run_host() {
    local work=$1
    configure_local_environment "$work" || return 1
    local -a prefix_args=()
    if [ -f "$work/prefix/protocol.yaml" ]; then
        prefix_args+=(--prefix-config "$work/prefix/protocol.yaml")
        if [ -f "$work/phase_action_decoder_protocol.yaml" ]; then
            prefix_args+=(--evidence-table-after-prefix --lossless-observation-history --verification-message-replies --phase-action-decoder)
        elif [ -f "$work/verification_reply_protocol.yaml" ]; then
            prefix_args+=(--evidence-table-after-prefix --lossless-observation-history --verification-message-replies)
        elif [ -f "$work/observation_history_repair.yaml" ]; then
            prefix_args+=(--evidence-table-after-prefix --lossless-observation-history)
        else
            local arm
            IFS= read -r arm < "$work/diagnostic_arm.txt"
            if [ "$arm" = actor_local_evidence_table ]; then
                prefix_args+=(--evidence-table-after-prefix)
            elif [ "$arm" != original_view ]; then
                return 1
            fi
        fi
    fi
    if [ -f "$work/action_decoder_bounds.yaml" ]; then
        prefix_args+=(--action-decoder-bounds "$work/action_decoder_bounds.yaml")
    fi
    failure_phase=host_process
    run_and_wait "$APPWORLD_GPU_PYTHON" -u -m Libs.scripts.run_appworld_local_collaboration \
        "${prefix_args[@]}" --actor-config "$work/Libs/configs/neural_population_appworld_tool_actor.yaml" \
        --collaboration-config "$work/Libs/configs/neural_population_appworld_local_collaboration.yaml" \
        --allocation "$work/allocation.yaml" --api-schemas "$work/api_docs.json" \
        --metadata "$work/model_metadata.json" --model-directory "$MODEL_DIRECTORY" \
        --controller-command-json "$work/controller_command.json" --output "$work/host" \
        >"$work/host_stdout.log" 2>"$work/host_stderr.log"
    local code=$?
    failure_phase=process_receipt
    helper "$APPWORLD_GPU_PYTHON" record_exit "$work" "$work" gpu_host "$code"
    local receipt_code=$?
    if [ "$receipt_code" -ne 0 ]; then return 1; fi
    if [ "$code" -eq 2 ]; then failure_armed=0; fi
    return "$code"
}

case "${1-}" in
    __stage_cpu)
        install_failure_handler "$2" "$3" cpu_controller cpu_stage "$CPU_PYTHON"
        failure_phase=stage_cpu
        helper "$CPU_PYTHON" stage_cpu "$2" "$3" cpu_controller
        code=$?
        if [ "$code" -eq 0 ]; then failure_armed=0; fi
        exit "$code"
        ;;
    __native)
        install_failure_handler "$2" "$(run_archive_path)" cpu_controller cpu_controller "$CPU_PYTHON"
        run_native "$2"
        exit "$?"
        ;;
    __host)
        install_failure_handler "$2" "$(run_archive_path)" gpu_host gpu_host "$APPWORLD_GPU_PYTHON"
        run_host "$2"
        exit "$?"
        ;;
    __archive_cpu)
        install_failure_handler "$2" "$3" cpu_controller cpu_archive "$CPU_PYTHON"
        failure_phase=archive_cpu
        helper "$CPU_PYTHON" archive "$2" "$3" cpu_controller "$4"
        exit "$?"
        ;;
esac

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[012]$ ]]; then
    printf '%s\n' 'Usage: sbatch appworld_local_collaboration.sbatch PLANNER_QUOTA (0, 1, or 2)' >&2
    exit 1
fi
: "${APPWORLD_GPU_PYTHON:?Export the GPU model environment Python path}"
: "${APPWORLD_MODEL_METADATA:?Export the exact pinned official metadata JSON path}"
: "${APPWORLD_READINESS_REPORT:?Export the successful pinned readiness report JSON path}"
if [[ "$APPWORLD_GPU_PYTHON" != /* ]] || [ ! -x "$APPWORLD_GPU_PYTHON" ]; then
    printf '%s\n' 'APPWORLD_GPU_PYTHON must be an executable absolute path on the selected GPU node' >&2
    exit 1
fi
if [ -n "${APPWORLD_PHASE_ACTION_DECODER_PROTOCOL:-}" ] && [ -z "${APPWORLD_LAUNCHER_SOURCE:-}" ]; then
    printf '%s\n' 'APPWORLD_LAUNCHER_SOURCE must name the exact validated launcher for decoder execution' >&2
    exit 1
fi
if [[ "$LAUNCHER_SOURCE" != /* ]] || [ ! -f "$LAUNCHER_SOURCE" ]; then
    printf '%s\n' 'APPWORLD_LAUNCHER_SOURCE must be an existing absolute file path' >&2
    exit 1
fi
job_id=${SLURM_HET_JOB_ID:-${SLURM_JOB_ID:-}}
if [[ ! "$job_id" =~ ^[0-9]+$ ]] || [ "${SLURM_HET_SIZE:-}" != 2 ]; then
    printf '%s\n' 'A two-component heterogeneous Slurm allocation is required' >&2
    exit 1
fi
export APPWORLD_HET_LEADER_JOB_ID="$job_id"
work="${APPWORLD_SCRATCH_ROOT:-/tmp/neuronsim}/neuronsim-appworld-local-$job_id"
archive="$REPO/Output/neural_population_dynamics/agentic_extension/appworld_local_collaboration/run_$job_id"
install_failure_handler "$work" "$archive" gpu_host launcher "$APPWORLD_GPU_PYTHON"
failure_phase=stage_gpu
helper "$APPWORLD_GPU_PYTHON" stage_gpu "$work" "$archive" gpu_host "$1" "$LAUNCHER_SOURCE" "$EXECUTION_LAUNCHER_SOURCE"
stage_code=$?
if [ "$stage_code" -ne 0 ]; then exit 1; fi

failure_phase=stage_cpu
run_and_wait srun --het-group=1 --cpu-bind=cores --ntasks=1 --cpus-per-task=1 --mem=2G --gres=none \
    bash "$archive/inputs/launcher.sbatch" __stage_cpu "$work" "$archive" >&2
cpu_stage_code=$?
case_code=1
cpu_archive_code=0
if [ "$cpu_stage_code" -eq 0 ]; then
    failure_phase=host_step
    run_and_wait srun --het-group=0 --cpu-bind=cores --ntasks=1 --cpus-per-task=4 --mem=128G --gres=gpu:2 \
        bash "$work/launcher.sbatch" __host "$work"
    case_code=$?
    if [ "$case_code" -ne 2 ]; then
        if [ "$case_code" -eq 0 ]; then exit 1; fi
        exit "$case_code"
    fi
    failure_phase=archive_cpu
    run_and_wait srun --het-group=1 --cpu-bind=cores --ntasks=1 --cpus-per-task=1 --mem=2G --gres=none \
        bash "$archive/inputs/launcher.sbatch" __archive_cpu "$work" "$archive" "$case_code" >&2
    cpu_archive_code=$?
    if [ "$cpu_archive_code" -ne 0 ]; then exit "$cpu_archive_code"; fi
fi
failure_phase=launcher_receipt
helper "$APPWORLD_GPU_PYTHON" record_launcher "$work" "$archive" gpu_host \
    "$cpu_stage_code" "$case_code" "$cpu_archive_code"
record_code=$?
failure_phase=archive_gpu
helper "$APPWORLD_GPU_PYTHON" archive "$work" "$archive" gpu_host "$case_code"
gpu_archive_code=$?
if [ "$cpu_stage_code" -eq 0 ] && [ "$case_code" -eq 2 ] && [ "$cpu_archive_code" -eq 0 ] \
        && [ "$record_code" -eq 0 ] && [ "$gpu_archive_code" -eq 0 ]; then
    failure_armed=0
    exit 0
fi
exit 1

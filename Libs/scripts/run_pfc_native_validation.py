"""Stage and run the authorized native PFC comparison in one CPU allocation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


MODULES = (
    'pfc_native_reference_execution', 'pfc_source_comparison', 'pfc_scale_analysis',
    'pfc_reference', 'pfc_reference_replay', 'pfc_delay_analysis',
    'pfc_response_analysis', 'pfc_kernel_parameters', 'pfc_factorized_source',
    'pfc_initial_rng', 'circuit_data', 'circuit_campaign', 'config', 'provenance',
    'resource_probe',
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_file(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    value = digest(target)
    if expected is not None:
        if target.stat().st_size != expected['bytes'] or value != expected['sha256']:
            raise ValueError('Declared staging input differs: ' + str(source))
    elif value != digest(source):
        raise ValueError('Staging copy differs: ' + str(source))
    return {'path': str(source), 'bytes': target.stat().st_size, 'sha256': value}


def stage_archive(source, target):
    delivery = json.loads((source / 'delivery.json').read_text())
    names = set()
    for row in delivery['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in names:
            raise ValueError('Invalid or duplicate delivered path')
        names.add(str(relative))
        copy_file(source / relative, target / relative, row)
    copy_file(source / 'delivery.json', target / 'delivery.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--resource-detector', type=Path, required=True,
                        help='Explicit delivered resource detector; staged to node-local scratch')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not args.resource_detector.is_absolute() or not args.resource_detector.is_file():
        parser.error('An existing absolute delivered resource detector is required')
    request = json.loads(args.request.read_text())
    root = Path(request['project_root'])
    output = args.request.parent / ('run_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    record = dict(status='PREPARING_AUTHORIZED_NATIVE_VALIDATION',
        job_id=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'],
        started_utc=datetime.now(timezone.utc).isoformat(),
        request=request, native_exit_code=None, comparison_exit_code=None,
        scientific_protocol_changed=False, production_episodes=0, phases=[])
    code = 1
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-validation-', dir='/tmp') as local:
        work = Path(local)
        record['scratch'] = str(work)
        try:
            if record['node'] != 'cpu03' or os.environ.get('SLURM_JOB_GPUS'):
                raise ValueError('CPU03 no-GPU allocation required')
            if shutil.disk_usage(work).free < 20 * 1024**3:
                raise ValueError('Node-local storage below required safety margin')
            package = work / 'Libs/population_dynamics'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            record['software'] = {}
            for name in MODULES:
                source = root / 'Libs/population_dynamics' / (name + '.py')
                record['software'][name] = copy_file(source, package / source.name)
            copy_file(root / 'Libs/scripts/run_pfc_source_cases.py',
                      work / 'Libs/scripts/run_pfc_source_cases.py')
            detector = work / 'detect_resources.py'
            record['resource_detector'] = copy_file(args.resource_detector, detector)
            runpy.run_path(str(detector))['detect_all_resources'](str(work / 'resources.json'))
            sys.path.insert(0, str(work / 'Libs'))
            from population_dynamics.resource_probe import probe
            resources = probe(work / 'resource_preflight.json', 8)
            if resources['status'] != 'PASS':
                raise ValueError('Node resources unavailable: ' + repr(resources['reasons']))
            env = dict(os.environ, PYTHONPATH=str(work / 'Libs'), TMPDIR=str(work),
                       PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')

            def run_phase(name, command):
                phase = dict(name=name, command=command,
                             started_utc=datetime.now(timezone.utc).isoformat())
                record['phases'].append(phase)
                record['status'] = name.upper()
                (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
                mark = time.monotonic()
                with (work / (name + '.log')).open('w') as log:
                    process = subprocess.run(command, env=env, cwd=work,
                                             stdout=log, stderr=subprocess.STDOUT)
                phase.update(exit_code=process.returncode, wall_seconds=time.monotonic()-mark,
                             completed_utc=datetime.now(timezone.utc).isoformat())
                copy_file(work / (name + '.log'), output / (name + '.log'))
                return process.returncode

            native = request['native_command']
            preflight = [argument for argument in native if argument != '--execute']
            preflight.append('--preflight-only')
            preflight_exit = run_phase('native_preflight', preflight)
            if preflight_exit:
                raise RuntimeError('Native preflight failed: ' + str(preflight_exit))
            if args.preflight_only:
                record['status'] = 'PREFLIGHT_COMPLETE_NATIVE_SIMULATION_NOT_EXECUTED'
                code = 0
                return code
            record['native_exit_code'] = run_phase('native_validation', native)
            if record['native_exit_code']:
                raise RuntimeError('Native validation failed: ' + str(record['native_exit_code']))
            native_root = Path(native[native.index('--output') + 1])
            delivery = json.loads((native_root / 'delivery.json').read_text())
            if delivery['status'] != 'NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE' or delivery['exit_code']:
                raise ValueError('Native terminal delivery is not complete')
            record['native_delivery_sha256'] = digest(native_root / 'delivery.json')
            compare = list(request['comparison_command'])
            native_local = work / 'native_reference'
            source_positions = [index + 1 for index, argument in enumerate(compare)
                                if argument == '--source-run']
            original_sources = [Path(compare[position]) for position in source_positions]
            source_archives = [source.parent if source.name == 'source_run' else source
                               for source in original_sources]
            source_locals = [work / ('candidate_source' if index == 0 else
                                     'candidate_source_' + str(index + 1))
                             for index in range(len(original_sources))]
            record['input_origins'] = dict(native=str(native_root), candidate=str(source_archives[0]),
                candidates=[dict(source_run=str(source), archive=str(archive),
                                 staged_source_run=str(local / 'source_run'))
                            for source, archive, local in zip(original_sources, source_archives, source_locals)])
            stage_archive(native_root, native_local)
            for position, archive, local in zip(source_positions, source_archives, source_locals):
                stage_archive(archive, local)
                compare[position] = str(local / 'source_run')
            proof_root = Path(compare[compare.index('--factorization-proof') + 1])
            proof_local = work / 'factorization_proof'
            for name in ('source_comparison.json', 'factorized/report.json', 'design.json',
                         'Libs/population_dynamics/pfc_factorized_source.py'):
                copy_file(proof_root / name, proof_local / name)
            design = Path(compare[compare.index('--design') + 1])
            copy_file(design, work / 'preparation/design.json')
            copy_file(design.parent / 'native_config.yaml', work / 'preparation/native_config.yaml')
            compare[compare.index('--reference') + 1] = str(native_local / 'native')
            compare[compare.index('--config') + 1] = str(source_locals[0] / 'source_run/config.yaml')
            compare[compare.index('--factorization-proof') + 1] = str(proof_local)
            compare[compare.index('--design') + 1] = str(work / 'preparation/design.json')
            compare[compare.index('--output') + 1] = str(work / 'comparison/report.json')
            record['input_origins'].update(factorization_proof=str(proof_root), design=str(design))
            record['comparison_exit_code'] = run_phase('source_comparison', compare)
            local_report = work / 'comparison/report.json'
            if local_report.is_file():
                report = json.loads(local_report.read_text())
                record['comparison_status'] = report['status']
                record['comparison_report'] = request['comparison_output']
                target = Path(request['comparison_output'])
                if target.exists():
                    raise FileExistsError(target)
                copy_file(local_report, target)
            code = record['comparison_exit_code']
            record['status'] = ('COMPLETE_AUTHORIZED_NATIVE_AND_COMPARISON'
                                if code == 0 else 'COMPLETE_COMPARISON_WITH_NUMERICAL_FAILURE')
        except Exception as error:
            record.update(status='AUTHORIZED_VALIDATION_FAILED', error=str(error),
                          error_type=type(error).__name__, traceback=traceback.format_exc())
            code = record['native_exit_code'] or record['comparison_exit_code'] or 1
        finally:
            for name in ('resources.json', 'resource_preflight.json'):
                if (work / name).is_file():
                    copy_file(work / name, output / name)
            # Preserve the executed code for interpreting the retained reports.
            for name in MODULES:
                source = work / 'Libs/population_dynamics' / (name + '.py')
                if source.is_file():
                    copy_file(source, output / 'software/population_dynamics' / source.name)
            record.update(exit_code=code, wall_seconds=time.monotonic()-started,
                          completed_utc=datetime.now(timezone.utc).isoformat())
            (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

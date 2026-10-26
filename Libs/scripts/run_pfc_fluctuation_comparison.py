"""Stage and run the PFC development fluctuation comparison inside Slurm."""
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
import traceback


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    job = os.environ['SLURM_JOB_ID']
    if os.environ.get('SLURMD_NODENAME') not in ('cpu04', 'cpu05') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Allowed CPU-only Slurm placement required')
    base = repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/streamed_source_execution'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock-archive', action='append', type=Path,
                        help='Delivered complete source clock archive; provide exactly three')
    parser.add_argument('--native-reference', type=Path,
                        default=base/'reference_fluctuation_calibration/run_586261')
    parser.add_argument('--config', type=Path,
                        default=repo/'Libs/configs/neural_population_pfc_fluctuation_comparison.yaml')
    parser.add_argument('--dependency-archive', type=Path,
                        default=base/'fluctuation_calibration/run_586418')
    parser.add_argument('--precision-config', type=Path)
    parser.add_argument('--validate-only', action='store_true', help='Validate software without reprocessing source archives')
    args = parser.parse_args(argv)
    clocks = args.clock_archive if args.clock_archive is not None else [
        base/'fluctuation_calibration'/('run_'+str(number)) for number in (586427,586418,586428)]
    if len(clocks) != 3:
        parser.error('exactly three --clock-archive arguments required')
    target = base/'fluctuation_comparison'/('run_'+job)
    if target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-fluctuation-comparison-'+job+'.', dir='/tmp') as temporary:
        work = Path(temporary)
        for name in ('scratch', 'phases'):
            (work/name).mkdir()
        record = dict(status='PREPARING', job_id=job, node=os.environ['SLURMD_NODENAME'],
            started_utc=datetime.now(timezone.utc).isoformat(), commands=[], new_source_simulations=0,
            production_episodes=0, scientific_fidelity_qualified=False)
        code = 1
        try:
            prior = args.dependency_archive.resolve()
            prior_delivery = json.loads((prior/'delivery.json').read_text())
            if prior_delivery.get('exit_code') != 0:
                raise ValueError('Completed dependency archive required')
            names = set()
            for row in prior_delivery['files']:
                name = Path(row['path'])
                if name.is_absolute() or '..' in name.parts or str(name) in names:
                    raise ValueError('Invalid prior archive identity')
                names.add(str(name))
                if not (str(name).startswith('Libs/') or str(name) == 'run_guarded.py'):
                    continue
                destination = work/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prior/name, destination)
                if destination.stat().st_size != row['bytes'] or digest(destination) != row['sha256']:
                    raise ValueError('Changed verified dependency: '+str(name))
            record['dependencies'] = dict(archive=str(prior), delivery_sha256=digest(prior/'delivery.json'))
            for name in ('pfc_fluctuation_comparison.py', 'test_pfc_fluctuation_comparison.py',
                         'pfc_numerical_precision.py', 'test_pfc_numerical_precision.py',
                         'pfc_phase_responses.py', 'test_pfc_phase_responses.py'):
                shutil.copy2(repo/'Libs/population_dynamics'/name, work/'Libs/population_dynamics'/name)
            shutil.copy2(args.config, work/'config.yaml')
            shutil.copy2(repo/'Libs/configs/neural_population_pfc_numerical_precision.yaml',
                         work/'Libs/configs/neural_population_pfc_numerical_precision.yaml')
            if args.precision_config is not None:
                shutil.copy2(args.precision_config,work/'precision_config.yaml')
            shutil.copy2(Path(__file__), work/'launcher.py')
            (work/'run_tests.py').write_text("""import json,unittest
from pathlib import Path
suite=unittest.defaultTestLoader.loadTestsFromNames(['population_dynamics.test_pfc_fluctuation_comparison','population_dynamics.test_pfc_numerical_precision','population_dynamics.test_pfc_phase_responses'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
Path('tests.json').write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),success=result.wasSuccessful())))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
""")
            env = dict(os.environ, PYTHONPATH=str(work/'Libs'), TMPDIR=str(work/'scratch'),
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            def run(label, command):
                row = dict(label=label, command=[str(x) for x in command])
                record['commands'].append(row)
                result = subprocess.run([sys.executable, '-B', str(work/'run_guarded.py'), label, *row['command']],
                    cwd=work, env=env)
                row['exit_code'] = result.returncode
                if result.returncode:
                    raise RuntimeError(label+' failed: '+str(result.returncode))
            run('tests', [sys.executable, '-B', str(work/'run_tests.py')])
            record['tests'] = json.loads((work/'tests.json').read_text())
            if args.validate_only:
                record.update(status='VALIDATED_PFC_FLUCTUATION_COMPARISON',actual_calibration_analysis_executed=False)
            else:
                sources = [(f'clock_{index}', path.resolve()) for index,path in enumerate(clocks)]
                sources.append(('native_2560', args.native_reference.resolve()))
                origins = []
                command = [sys.executable, '-B', '-m', 'population_dynamics.pfc_fluctuation_comparison',
                    '--config', str(work/'config.yaml'), '--output', str(work/'analysis')]
                if args.precision_config is not None:
                    command += ['--precision-config',str(work/'precision_config.yaml')]
                for label, source in sources:
                    delivery = json.loads((source/'delivery.json').read_text())
                    if delivery.get('exit_code') != 0:
                        raise ValueError('Completed input archive required: '+str(source))
                    destination = work/'inputs'/label
                    names = ['delivery.json', 'config.yaml', 'analysis/report.json', 'analysis/unit_moments.npz',
                             'analysis/field_fluctuations.npz', 'analysis/population_variance.csv']
                    if any(row['path'] == 'analysis/unit_count_histograms.npz' for row in delivery['files']):
                        names.append('analysis/unit_count_histograms.npz')
                    if label != 'native_2560' and any(row['path'].startswith('reference/') for row in delivery['files']):
                        names += ['reference/'+name for name in ('delivery.json','analysis/analysis.json',
                            'analysis/readout_replay.json','analysis/responses.csv','analysis/phase_unit_counts.npz')]
                    for name in names:
                        output = destination/name
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source/name, output)
                        if name != 'delivery.json':
                            entries = [row for row in delivery['files'] if row['path'] == name]
                            if len(entries) != 1 or output.stat().st_size != entries[0]['bytes'] or digest(output) != entries[0]['sha256']:
                                raise ValueError('Changed delivered input: '+str(source/name))
                    origins.append(dict(cohort=label, original_archive=str(source),
                        staged=str(destination), delivery_sha256=digest(destination/'delivery.json')))
                    command += ['--native-reference' if label == 'native_2560' else '--clock-archive', str(destination)]
                (work/'input_origins.json').write_text(json.dumps(origins, indent=2)+'\n')
                run('comparison', command)
                result = json.loads((work/'analysis/report.json').read_text())
                if result['status'] != 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON':
                    raise ValueError('Real comparison did not complete')
                record.update(status=result['status'], row_counts=result['row_counts'],
                              source_episodes_reused=result['source_episodes_reused'],
                              undefined_phase_lag_rows=result['undefined_phase_lag_rows'],
                              count_cdf_comparison=result.get('count_cdf_comparison'),
                              calibration_precision=result.get('calibration_precision'),
                              source_phase_responses=result.get('source_phase_responses'),
                              registered_cases_by_cohort=result['registered_cases_by_cohort'],
                              independent_joint_blocks_by_cohort=result['independent_joint_blocks_by_cohort'])
            code = 0
        except Exception as error:
            record.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
        if (work/'tests.json').exists():
            record['tests'] = json.loads((work/'tests.json').read_text())
        record.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
        (work/'report.json').write_text(json.dumps(record, indent=2)+'\n')
        target.mkdir(parents=True, exist_ok=False)
        files = []
        for source in sorted(work.rglob('*')):
            relative = source.relative_to(work)
            if not source.is_file() or set(relative.parts)&{'scratch', '__pycache__'}:
                continue
            # Large inputs remain in their immutable original archives, bound by input_origins.json.
            if relative.parts[0] == 'inputs' and source.suffix in ('.npz', '.csv'):
                continue
            destination = target/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            checksum = digest(source)
            if digest(destination) != checksum:
                raise ValueError('Comparison delivery mismatch: '+str(relative))
            files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
        (target/'delivery.json').write_text(json.dumps(dict(record, files=files), indent=2)+'\n')
        print(json.dumps({key: record.get(key) for key in ('status', 'job_id', 'exit_code', 'tests', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())

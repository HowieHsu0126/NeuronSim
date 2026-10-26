"""Stage accepted saved ensembles, test the numerical path, and describe covariance.

CPU-only Slurm execution. Producer reports, delivery manifests and independent
replay receipts bind inputs. All large reads and calculations use local scratch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_population_trajectory_campaign import archive, copy_bound, digest, manifest_rows


def stage_ensemble(root: Path, replay: Path, work: Path, *, native: bool) -> dict:
    proof_path = root / 'delivery.json'
    proof = json.loads(proof_path.read_text())
    accepted = json.loads(replay.read_text())
    if (proof.get('status') != 'PASS' or proof.get('exit_code') != 0 or proof.get('scientific_claim_qualified') is not False
            or accepted.get('status') != 'PASS' or accepted.get('delivery_sha256') != digest(proof_path)
            or accepted.get('scientific_claim_qualified') is not False):
        raise ValueError('complete independently replayed development delivery required')
    key = 'exact_unit_count_values' if native else 'exact_generated_unit_count_values'
    if accepted.get(key) != 160000000:
        raise ValueError('complete160million count replay required')
    prefix = Path('analysis' if native else 'cost_point')
    report_name = 'analysis.json' if native else 'cost_point.json'
    index = manifest_rows(proof)
    names = [prefix / name for name in (report_name, 'sampled_unit_counts.npy', 'ensemble_fields.npz')]
    destination = work / 'inputs' / ('native' if native else 'learned')
    for name in names:
        copy_bound(root / name, destination / name.name, index[str(name)])
    copy_bound(proof_path, destination / 'delivery.json', dict(bytes=proof_path.stat().st_size, sha256=digest(proof_path)))
    copy_bound(replay, destination / 'replay.json', dict(bytes=replay.stat().st_size, sha256=digest(replay)))
    report = json.loads((destination / report_name).read_text())
    if native and accepted.get('analysis_sha256') != digest(destination / report_name):
        raise ValueError('native replay report binding changed')
    for name in ('sampled_unit_counts.npy', 'ensemble_fields.npz'):
        if report['artifacts'][name] != index[str(prefix / name)]['sha256']:
            raise ValueError('producer report and delivered sample identity differ')
    return dict(producer=str(root), delivery_sha256=digest(proof_path), replay=str(replay),
                replay_sha256=digest(replay), report_sha256=digest(destination / report_name))


def stage_retained_cell(root: Path, seed: int, work: Path) -> dict:
    proof_path = root / 'delivery.json'
    proof = json.loads(proof_path.read_text())
    if (proof.get('status') != 'PASS' or proof.get('exit_code') != 0
            or proof.get('training_seed') != seed or proof.get('scientific_claim_qualified') is not False
            or proof.get('calibration_or_test_accessed') is not False):
        raise ValueError('completed development-only population cell required')
    index = manifest_rows(proof)
    destination = work / 'inputs' / f'seed_{seed}'
    names = ('training/run.json', 'forecasts/evaluation.json', 'forecasts/full_unit_samples.npz', 'replay.json')
    for name in names:
        copy_bound(root / name, destination / name, index[name])
    copy_bound(proof_path, destination / 'delivery.json', dict(bytes=proof_path.stat().st_size, sha256=digest(proof_path)))
    return dict(seed=seed, producer=str(root), delivery_sha256=digest(proof_path),
                sample_sha256=index['forecasts/full_unit_samples.npz']['sha256'])


def stage_completed_analysis(root: Path, campaign: Path, work: Path) -> tuple[list[int], dict]:
    proof = json.loads((root / 'delivery.json').read_text())
    acceptance = json.loads((root / 'acceptance.json').read_text())
    if (proof.get('status') != 'DESCRIBED_RETAINED_POPULATION_COVARIANCE_DEVELOPMENT'
            or proof.get('exit_code') != 0 or acceptance.get('status') != 'PASS'
            or acceptance['delivery_sha256'] != digest(root / 'delivery.json')
            or proof.get('scientific_claim_qualified') is not False):
        raise ValueError('accepted retained analysis required')
    rows = manifest_rows(proof)
    for name in ('analysis/analysis.json', 'analysis/retained_unit_covariance.csv',
                 'analysis/delete_pair_sensitivity.csv', 'analysis/covariance_kernels.npz'):
        copy_bound(root / name, work / 'inputs/reused' / name, rows[name])
    # Reuse completed computation only while original delivered inputs match.
    seeds = []
    for cell in proof['inputs']:
        seed = cell['seed']
        delivery_path = campaign / f'seed_{seed}/delivery.json'
        if seed not in (41, 42, 43) or seed in seeds or digest(delivery_path) != cell['delivery_sha256']:
            raise ValueError('reused retained cell input changed')
        seeds.append(seed)
    copy_bound(root / 'delivery.json', work / 'inputs/reused/delivery.json',
               dict(bytes=(root / 'delivery.json').stat().st_size, sha256=digest(root / 'delivery.json')))
    return seeds, dict(input=str(root), delivery_sha256=digest(root / 'delivery.json'), reused_seeds=seeds)


def run(args) -> int:
    node, partition = os.environ.get('SLURMD_NODENAME'), os.environ.get('SLURM_JOB_PARTITION')
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS')
            or (node, partition) != ('cpu04', 'cpu') or Path('/tmp/neuronsim') not in args.work.parents):
        raise ValueError('CPU04 CPU-only Slurm allocation and node-local scratch required')
    if args.output.exists():
        raise FileExistsError(args.output)
    record = dict(status='RUNNING', exit_code=None, job_id=os.environ['SLURM_JOB_ID'], node=node,
                  command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_claim_qualified=False, generation_performed=False, training_performed=False,
                  simulation_performed=False, formal_test_accessed=False)
    try:
        names = ['Libs/population_dynamics/' + name for name in
                 ('__init__.py', 'provenance.py', 'resource_probe.py', 'unit_covariance_analysis.py', 'test_unit_covariance_analysis.py')]
        names += ['Libs/scripts/run_population_trajectory_campaign.py', 'Libs/scripts/run_unit_covariance_analysis.py',
                  'Libs/scripts/run_unit_covariance_analysis.sbatch']
        software_rows = None
        if args.software:
            proof = json.loads((args.software / 'delivery.json').read_text())
            accepted = json.loads((args.software / 'acceptance.json').read_text())
            if (proof['tests_exit_code'] != 0 or proof['exit_code'] != 0 or accepted['status'] != 'PASS'
                    or accepted['delivery_sha256'] != digest(args.software / 'delivery.json')
                    or accepted['tests_passed'] < 12):
                raise ValueError('accepted retained covariance software required')
            software_rows = manifest_rows(proof)
            record['software_delivery_sha256'] = digest(args.software / 'delivery.json')
        for name in names:
            source = (args.software or args.repo) / name
            copy_bound(source, args.work / name, software_rows[name] if software_rows else
                       dict(bytes=source.stat().st_size, sha256=digest(source)))
        sys.path[:0] = [str(args.work / 'Libs'), str(args.work)]
        from population_dynamics.resource_probe import probe
        if probe(args.work / 'admission.json', required_memory_gib=2)['status'] != 'PASS':
            raise RuntimeError('saved covariance resource admission failed')
        environment = dict(os.environ, PYTHONPATH=str(args.work / 'Libs'), PYTHONDONTWRITEBYTECODE='1',
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if software_rows:
            record.update(tests_exit_code=0, unchanged_software_tests_reused=True,
                          tests_passed=accepted['tests_passed'])
        else:
            tests = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                     'Libs/population_dynamics/test_unit_covariance_analysis.py']
            with (args.work / 'tests.log').open('w') as stream:
                result = subprocess.run(tests, cwd=args.work, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            record['tests_exit_code'] = result.returncode
            if result.returncode:
                raise RuntimeError('saved covariance numerical/staging tests failed')
        command = [sys.executable, '-m', 'population_dynamics.unit_covariance_analysis']
        if args.population_campaign:
            seeds = [args.seed] if args.seed else [41, 42, 43]
            record['training_seeds'] = seeds
            record['complete_three_seed_coverage'] = seeds == [41, 42, 43]
            reused_seeds = []
            if args.reuse_analysis:
                reused_seeds, record['reused_analysis'] = stage_completed_analysis(args.reuse_analysis, args.population_campaign, args.work)
                if not set(reused_seeds) <= set(seeds):
                    raise ValueError('reused seeds outside requested coverage')
                command += ['--previous-analysis', 'inputs/reused/analysis']
            record['inputs'] = [stage_retained_cell(args.population_campaign / f'seed_{seed}', seed, args.work)
                                for seed in seeds if seed not in reused_seeds]
            for seed in seeds:
                if seed in reused_seeds:
                    continue
                command += ['--retained-cell', f'inputs/seed_{seed}']
        else:
            record['inputs'] = dict(native=stage_ensemble(args.native, args.native_replay, args.work, native=True),
                                   learned=stage_ensemble(args.learned, args.learned_replay, args.work, native=False))
            command += ['--native', 'inputs/native', '--learned', 'inputs/learned']
        command += ['--output', 'analysis']
        with (args.work / 'analysis.log').open('w') as stream:
            result = subprocess.run(command, cwd=args.work, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record['analysis_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError('saved covariance analysis CLI failed')
        report = json.loads((args.work / 'analysis/analysis.json').read_text())
        if args.population_campaign:
            if sorted(cell['seed'] for cell in report['cells']) != seeds:
                raise ValueError('incomplete or duplicate retained output seed coverage')
        record.update(status=report['status'], exit_code=0, analysis_sha256=digest(args.work / 'analysis/analysis.json'),
                      summary=report['summary'], conditioning_matched=False)
    except Exception as error:
        record.update(status='FAILED', exit_code=1, error=str(error), traceback=traceback.format_exc())
    archive(args.work, args.output, record)
    (args.work / 'archive_complete').write_text(str(args.output))
    return record['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'work', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--native', type=Path)
    mode.add_argument('--population-campaign', type=Path)
    for name in ('native-replay', 'learned', 'learned-replay'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--seed', type=int, choices=(41, 42, 43), help='one completed-cell preparation; default all three seeds')
    parser.add_argument('--reuse-analysis', type=Path, help='accepted completed-cell result to reuse without count reads')
    parser.add_argument('--software', type=Path, help='accepted numerical/caller software snapshot')
    args = parser.parse_args()
    if args.native:
        if args.seed or args.reuse_analysis or any(getattr(args, name) is None for name in ('native_replay', 'learned', 'learned_replay')):
            parser.error('native mode requires both replay receipts and learned input, without seed selection')
    elif any(getattr(args, name) is not None for name in ('native_replay', 'learned', 'learned_replay')):
        parser.error('retained campaign mode does not consume a native comparison')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()

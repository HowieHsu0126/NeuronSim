#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_MultiSpikeReadback
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-multispike-readback-%j.out
#SBATCH --error=/tmp/neuronsim-multispike-readback-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_READBACK_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_READBACK'
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import traceback

import numpy as np

assert socket.gethostname().split('.')[0] == 'cpu04'
assert not os.environ.get('SLURM_JOB_GPUS')
repo = Path(os.environ['NEURONSIM_PROJECT_ROOT'])
source = Path(os.environ['NEURONSIM_MULTISPIKE_EXECUTION_ARCHIVE'])
expected = os.environ['NEURONSIM_MULTISPIKE_EXECUTION_DELIVERY_SHA256']
job = os.environ['SLURM_JOB_ID']
archive = repo / 'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/multispike_compensator/readback' / ('run_' + job)
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix='neuronsim-multispike-readback-' + job + '.', dir='/tmp'))
result = dict(status='FAILED_PFC_MULTISPIKE_READBACK', job_id=job, source_archive=str(source),
              source_delivery_sha256=expected, new_source_replays=0, new_independent_blocks=0,
              scientific_fidelity_qualified=False, source_probability_coverage_qualified=False,
              confirmation_execution_released=False, production_episodes=0)
delivered = False
exit_code = 1

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    raw = (source / 'delivery.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    delivery = json.loads(raw)
    assert delivery['verified'] is True and delivery['status'] == 'COMPLETE_PFC_MULTISPIKE_EXECUTE'
    assert len(delivery['files']) <= 100 and sum(x['bytes'] for x in delivery['files']) < 10 * 1024**2
    for entry in delivery['files']:
        relative = Path(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        assert target.stat().st_size == entry['bytes'] and digest(target) == entry['sha256']
    sys.path.insert(0, str(work))
    from Libs.population_dynamics.pfc_multispike_compensator import integrate_histogram
    wrapper = json.loads((work / 'report.json').read_text())
    report = json.loads((work / 'analysis/report.json').read_text())
    plan = json.loads((work / 'prepared/plan.json').read_text())
    assert wrapper['exit_code'] == 0 and wrapper['error'] is None
    assert report['status'] == 'COMPLETE_PFC_MULTISPIKE_READOUT'
    assert report['all_source_arrays_bitwise_equal'] and all(report['source_array_checks'].values())
    assert report['observed_ticks'] == report['partial_observed_ticks'] == 100000
    assert report['source_units'] == 100000 and report['source_duration_ms'] == 5000.0
    assert report['prepared_plan_sha256'] == digest(work / 'prepared/plan.json')
    assert report['law'] == plan['law']
    with (work / 'analysis/factorial_compensator.csv').open() as stream:
        table = list(csv.DictReader(stream))
    with np.load(work / 'analysis/weighted_input_responses.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    phases = list(arrays['phase_names']); fields = list(arrays['field_indices'])
    assert phases == plan['settings']['phase_names'] and fields == plan['settings']['fields']
    assert np.array_equal(arrays['input_counts'], np.arange(report['law']['cutoff'] + 1))
    assert arrays['success_weights'].shape == (len(phases), len(fields), report['law']['cutoff'] + 1)
    assert len(table) == len(phases) * len(fields) == 25
    indexed = {(row['phase'], int(row['field_index'])): row for row in table}
    assert len(indexed) == 25
    rows = []
    scale = int(report['law']['scale'])
    for pi, phase in enumerate(phases):
        for fi, field in enumerate(fields):
            row = indexed[(str(phase), int(field))]
            weights = arrays['success_weights'][pi, fi]
            total = int(arrays['eligible_weights'][pi, fi])
            observed = int(arrays['observed_factorial'][pi, fi])
            lo, hi, omitted = integrate_histogram(weights, total, report['law'])
            denominator = int(row['denominator'])
            assert row['source_path_complete'] == 'True'
            assert int(row['observed_complete_windows']) == int(row['declared_windows'])
            assert int(row['field_units']) == int(arrays['field_units'][fi])
            assert denominator == int(row['field_units']) * int(row['declared_windows'])
            assert int(row['eligible_previous_count_weight']) == total
            assert int(row['observed_factorial_sum']) == observed
            assert int(row['compensator_sum_lower_scaled']) == lo
            assert int(row['compensator_sum_upper_scaled']) == hi
            assert int(row['omitted_contribution_upper_scaled']) == omitted
            assert float(row['factorial_compensator_lower']) == lo / (scale * denominator)
            assert float(row['factorial_compensator_upper']) == hi / (scale * denominator)
            assert float(row['conditional_input_interval_width']) == (hi - lo) / (scale * denominator)
            positive = bool(np.any(weights > 0))
            classification = ('POSITIVE_RETAINED_CONDITIONAL_SIGNAL' if lo > 0 else
                              'ZERO_RETAINED_RESPONSE_OMITTED_INPUT_UNRESOLVED' if hi > 0 else
                              'ZERO_PREDICTABLE_WEIGHT' if total == 0 else 'EXACT_ZERO_CONDITIONAL_READOUT')
            rows.append(dict(phase=str(phase), field_index=int(field), denominator=denominator,
                observed_factorial_sum=observed, eligible_previous_count_weight=total,
                retained_success_weight_sum=int(weights.sum()), retained_nonzero_response_counts=int(np.count_nonzero(weights)),
                lower=lo / (scale * denominator), upper=hi / (scale * denominator),
                omitted_upper=omitted / (scale * denominator), classification=classification))
    result.update(status='COMPLETE_PFC_MULTISPIKE_READBACK', source_array_checks=len(report['source_array_checks']),
        verified_files=len(delivery['files']), verified_bytes=sum(x['bytes'] for x in delivery['files']),
        source_analysis_report_sha256=digest(work / 'analysis/report.json'),
        reused_tests_archive=wrapper['commands'][0]['archive'], reused_tests_passed=6,
        rows=rows, positive_lower_rows=sum(row['lower'] > 0 for row in rows),
        zero_lower_positive_upper_rows=sum(row['lower'] == 0 < row['upper'] for row in rows),
        exact_zero_rows=sum(row['upper'] == 0 for row in rows),
        observed_factorial_total=sum(row['observed_factorial_sum'] for row in rows),
        total_eligible_weight=sum(row['eligible_previous_count_weight'] for row in rows),
        conditional_transition_parity_units=report['conditional_transition_parity_units'],
        observed_ticks=report['observed_ticks'], source_units=report['source_units'],
        source_rng_seed=report['source_rng_seed'], source_case_id=report['source_case_id'],
        observer_seconds=report['observer_seconds'], simulation_seconds=report['simulation_cost']['wall_seconds'],
        source_execute_seconds=wrapper['commands'][-1]['wall_seconds'],
        interpretation='Intervals enclose conditional compensators only on this one saved history. No population confidence or source-law tail qualification follows; rows contain no additional independent blocks.')
    exit_code = 0
except Exception as exc:
    result['error'] = dict(type=type(exc).__name__, message=str(exc))
    traceback.print_exc()
finally:
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        archive.mkdir(parents=True, exist_ok=False)
        (archive / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        shutil.copy2(Path(os.environ['NEURONSIM_READBACK_LAUNCHER']), archive / 'launcher.sbatch')
        if 'rows' in result:
            with (archive / 'readout_summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
                writer.writeheader(); writer.writerows(result['rows'])
        files = [dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(archive.iterdir()) if p.is_file()]
        (archive / 'delivery.json').write_text(json.dumps(dict(status=result['status'], verified=True, files=files), indent=2) + '\n')
        delivered = True
        print(json.dumps(result))
    finally:
        if delivered:
            shutil.rmtree(work)
        else:
            print('COPYBACK_FAILED_SCRATCH_PRESERVED:' + str(work), file=sys.stderr)
raise SystemExit(exit_code)
PY_READBACK

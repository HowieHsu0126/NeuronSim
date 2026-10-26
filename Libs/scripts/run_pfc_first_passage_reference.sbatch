#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_FirstPassageRef
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-first-passage-reference-%j.out
#SBATCH --error=/tmp/neuronsim-first-passage-reference-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_REFERENCE_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_REFERENCE'
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback

import numpy as np
import yaml

assert socket.gethostname().split('.')[0] == 'cpu03'
assert not os.environ.get('SLURM_JOB_GPUS')
repo = Path(os.environ['NEURONSIM_PROJECT_ROOT'])
assert (repo/'PROJECT_PREFIX').read_text().strip() == 'NrnSim'
job = os.environ['SLURM_JOB_ID']
family = Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/first_passage_reference')
archive = repo/family/'execution'/('run_'+job)
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix='neuronsim-first-passage-reference-'+job+'.', dir='/tmp'))
inputs = []
report = dict(status='FAILED_PFC_FIRST_PASSAGE_REFERENCE_VALIDATION', job_id=job,
              new_full_episodes=0, new_independent_blocks=0, first_passage_probability_established=False,
              population_bound_established=False, confirmation_execution_released=False,
              scientific_fidelity_qualified=False, production_episodes=0,
              suppressed_network_replay_implemented=False)
started = time.perf_counter(); delivered = False; exit_code = 1; simulation = None

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def copy(source, relative, expected=None):
    target = work/relative; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    actual = digest(target)
    if expected is not None and actual != expected:
        raise ValueError('FIRST_PASSAGE_INPUT_CHANGED:'+str(relative))
    inputs.append(dict(source=str(source), path=str(relative), sha256=actual))
    return target

def signature(array):
    array = np.asarray(array)
    assert array.flags.c_contiguous and np.isfinite(array).all()
    return (array.dtype.str, tuple(array.shape), hashlib.sha256(memoryview(array)).hexdigest())

try:
    config_relative = Path('Libs/configs/neural_population_pfc_first_passage_reference.yaml')
    config_path = copy(repo/config_relative, config_relative, os.environ['NEURONSIM_REFERENCE_CONFIG_SHA256'])
    settings = yaml.safe_load(config_path.read_text())
    assert settings['schema'] == 'pfc_first_passage_reference_validation_v1'
    prepared = repo/settings['preparation']['archive']
    copy(prepared/'delivery.json', Path('preparation_delivery.json'), settings['preparation']['delivery_sha256'])
    manifest = json.loads((work/'preparation_delivery.json').read_text())
    assert manifest['verified'] and manifest['status'] == 'COMPLETE_PFC_FIRST_PASSAGE_REFERENCE_PREPARATION'
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.parts[0] == 'Libs' or entry['path'] in ('report.json','tests.xml','tests.stdout'):
            target = Path('preparation_'+relative.name) if relative.parts[0] != 'Libs' else relative
            copy(prepared/relative, target, entry['sha256'])
    preparation = json.loads((work/'preparation_report.json').read_text())
    assert preparation['exit_code'] == 0
    assert sum(int(row['tests']) for row in preparation['test_suites']) == 9
    assert all(int(row['failures']) == int(row['errors']) == 0 for row in preparation['test_suites'])
    copy(Path(os.environ['NEURONSIM_REFERENCE_LAUNCHER']), Path('launcher.sbatch'))
    sys.path.insert(0, str(work))
    from Libs.population_dynamics import pfc_first_passage_reference as reference
    from Libs.population_dynamics import pfc_deterministic_replay as source
    from Libs.population_dynamics import pfc_native_drive as native
    from Libs.population_dynamics.pfc_multispike_compensator import observer_anchors
    from Libs.population_dynamics.pfc_fluctuation_analysis import unit_mapping
    from Libs.population_dynamics.pfc_delay_analysis import PHASES
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work/'resource_preflight.json', 4)
    assert health['status'] == 'PASS', health['reasons']
    original = repo/settings['source']['archive']
    copy(original/'delivery.json', Path('source_delivery.json'), settings['source']['delivery_sha256'])
    original_delivery = json.loads((work/'source_delivery.json').read_text())
    assert original_delivery['source_execution_complete'] and original_delivery['exit_code'] == 0
    entries = {row['path']: row for row in original_delivery['files']}
    for name in ('report.json','config.yaml','initial_voltage.npy','raw.npz'):
        relative = Path('source_run')/name
        copy(original/relative, Path('source_inputs')/name, entries[str(relative)]['sha256'])
    assert digest(work/'source_inputs/report.json') == settings['source']['report_sha256']
    for name in ('pfc_deterministic_replay.py','pfc_native_drive.py'):
        relative = Path('Libs/population_dynamics')/name
        assert digest(work/relative) == entries[str(relative)]['sha256']
    original_report = json.loads((work/'source_inputs/report.json').read_text())
    config = yaml.safe_load((work/'source_inputs/config.yaml').read_text())
    case = original_report['selected_case']
    assert config == original_report['resolved_config']
    assert case['case_id'] == settings['case_id'] and case['source_rng_seed'] == settings['source_rng_seed']
    assert config['dt_ms'] == settings['dt_ms'] and config['duration_ms'] == settings['source_duration_ms']
    assert config['excitatory_neurons'] + config['inhibitory_neurons'] == settings['expected_units'] == 100000
    with np.load(work/'source_inputs/raw.npz', allow_pickle=False) as data:
        raw = {key:data[key] for key in ('spike_ticks','spike_unit_indices','native_spike_counts','spike_seconds')}
    tape = reference.validate_tape(config, raw)
    del raw
    mapping, _ = unit_mapping(tape.n_e, tape.n_i, config['excitatory_preference_bins'])
    eligible = np.flatnonzero(mapping[:tape.n_e] == settings['field_index'])
    phase_start, phase_stop = PHASES[settings['phase']]
    selection = reference.select_first_spike_window(config, tape, eligible_units=eligible,
        phase_start_tick=round(phase_start/config['dt_ms']), phase_stop_tick=round(phase_stop/config['dt_ms']),
        window_ms=settings['window_ms'])
    report['selection'] = asdict(selection)
    # This branch validates one no-hit reference. A later hit is preserved as a failure,
    # never replaced by a different selected unit or by a zero probability.
    reference.require_observed_reference(tape, selection)
    stop_tick = selection.window_stop_tick
    expected = {}; thin = []
    def capture(stage, tick, snapshot):
        expected[(stage,tick)] = {key:signature(value) for key,value in snapshot.items()}
        if stage == 'pre_rk2' and selection.first_spike_tick < tick < stop_tick:
            prefix = 'E__'; u = selection.unit_index
            thin.append(dict(tick=tick, time_seconds=tick*tape.dt_seconds,
                **{field:float(snapshot[prefix+field][u]) for field in ('s_GABA','s_NMDA_total','I_stim','s_NMDA','x')}))
    initial, reconstruction_drive = native.create_drive(config, case)
    assert np.array_equal(initial, np.load(work/'source_inputs/initial_voltage.npy', allow_pickle=False))
    begin = time.perf_counter()
    reconstructed = reference.reconstruct(config, case, tape, stop_tick=stop_tick,
        drive=reconstruction_drive, selection=selection, on_snapshot=capture)
    report['reconstruction_seconds'] = time.perf_counter()-begin
    conditional = reconstructed['conditional_initial']
    checks = dict(snapshot_calls=0, array_digests=0, state_values=0, reset_state_equal=False)
    anchors = observer_anchors(); stages = {row['line']:stage for stage,row in anchors.items()}
    target = source.simulate.__code__; actual_first_input = None
    def trace(frame, event, arg):
        global actual_first_input
        if frame.f_code is not target:
            return None
        if event == 'line' and frame.f_lineno in stages:
            stage = stages[frame.f_lineno]; local = frame.f_locals; tick = int(local['tick'])
            planned = expected[(stage,tick)]
            for key, expected_signature in planned.items():
                population, field = key.split('__',1)
                value = local[population.lower()][field]
                if signature(value) != expected_signature:
                    np.save(work/'failed_source_state.npy', value)
                    report['first_mismatch'] = dict(stage=stage,tick=tick,field=key,expected_signature=expected_signature)
                    raise ValueError('FIRST_PASSAGE_REFERENCE_STATE_MISMATCH:'+key)
                checks['array_digests'] += 1; checks['state_values'] += int(value.size)
            checks['snapshot_calls'] += 1
            if stage == 'pre_input' and tick == selection.first_spike_tick:
                actual_first_input = float(local['drive']['E'][tick % local['chunk_rows'],selection.unit_index])
            if stage == 'pre_rk2' and tick == selection.first_spike_tick+1:
                e = local['e']; u = selection.unit_index
                assert actual_first_input is not None
                assert e['v'][u] == conditional['state']['v']
                assert e['lastspike'][u] == conditional['state']['lastspike']
                assert e['x'][u] == conditional['state']['x']
                assert e['s_AMPA'][u] == conditional['state']['s_AMPA'] + actual_first_input
                checks['reset_state_equal'] = True
        return trace
    prefix_config = dict(config, duration_ms=stop_tick*config['dt_ms'])
    source_initial, source_drive = native.create_drive(config, case)
    previous = sys.gettrace(); begin = time.perf_counter()
    try:
        sys.settrace(trace)
        simulation = source.simulate(prefix_config, case, source_initial, backend='numpy', drive=source_drive)
    finally:
        sys.settrace(previous)
    report['source_prefix_seconds'] = time.perf_counter()-begin
    events = reference.canonical_events(simulation['raw']['spike_ticks'], simulation['raw']['spike_unit_indices'],
                                        tape.n_e+tape.n_i, stop_tick)
    expected_events = tape.events[tape.events[:,0] < stop_tick]
    assert np.array_equal(events, expected_events)
    assert np.array_equal(simulation['raw']['native_spike_counts'], np.bincount(expected_events[:,1], minlength=tape.n_e+tape.n_i))
    terminal_checks = {key:np.array_equal(value,simulation['terminal_state'][key])
                       for key,value in reconstructed['terminal_state'].items()}
    assert all(terminal_checks.values())
    first_rng = native._state_arrays(reconstruction_drive.get_state())
    second_rng = native._state_arrays(source_drive.get_state())
    assert all(np.array_equal(first_rng[key],second_rng[key]) for key in first_rng)
    assert checks['snapshot_calls'] == stop_tick*3 and checks['reset_state_equal']
    assert len(thin) == stop_tick-selection.first_spike_tick-1
    np.savez_compressed(work/'reference_window.npz', **{key:np.asarray([row[key] for row in thin]) for key in thin[0]})
    (work/'conditional_initial.json').write_text(json.dumps(conditional,indent=2)+'\n')
    report.update(status='COMPLETE_PFC_FIRST_PASSAGE_REFERENCE_VALIDATION', checks=checks,
        all_snapshot_digests_equal=True, terminal_state_checks=terminal_checks, source_prefix_events_equal=True,
        source_prefix_events=int(len(events)), all_prefix_rng_state_equal=True,
        prefix_ticks=stop_tick, prefix_duration_ms=prefix_config['duration_ms'], source_units=tape.n_e+tape.n_i,
        source_case_id=case['case_id'], source_rng_seed=case['source_rng_seed'],
        conditional_initial=conditional, reference_window_rows=len(thin), reference_window_excludes_focal_future_ampa_and_voltage=True,
        reference_units=dict(tick='source_tick',time_seconds='s',s_GABA='dimensionless',s_NMDA_total='dimensionless',I_stim='A',s_NMDA='dimensionless',x='dimensionless'),
        membrane_integration_in_reconstruction=False, actual_source_prefix_replays=1,
        reused_preparation=str(prepared), reused_tests_passed=9,
        interpretation='One selected no-second-spike reference; source states reconstructed from complete bound spike tape. Pointwise and prefix checks do not establish first-passage probability or population uncertainty. Actual hit paths still require a suppressed-network replay.')
    exit_code = 0
except Exception as exc:
    report['error'] = dict(type=type(exc).__name__,message=str(exc)); traceback.print_exc()
    if simulation is not None:
        np.savez_compressed(work/'failed_source_raw.npz', **simulation['raw'])
        np.savez_compressed(work/'failed_source_terminal.npz', **simulation['terminal_state'])
finally:
    report.update(inputs=inputs, wall_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        archive.mkdir(parents=True,exist_ok=False); files=[]
        for path in sorted(work.rglob('*')):
            if not path.is_file(): continue
            relative = path.relative_to(work)
            if relative.parts[0] == 'source_inputs' or '__pycache__' in relative.parts: continue
            destination = archive/relative; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination); checksum=digest(path)
            assert digest(destination) == checksum
            files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=checksum))
        (archive/'delivery.json').write_text(json.dumps(dict(status=report['status'],verified=True,job_id=job,files=files),indent=2)+'\n')
        delivered=True; print(json.dumps(dict(status=report['status'],archive=str(archive),exit_code=exit_code)))
    finally:
        if delivered: shutil.rmtree(work)
        else: print('COPYBACK_FAILED_SCRATCH_PRESERVED:'+str(work),file=sys.stderr)
raise SystemExit(exit_code)
PY_REFERENCE

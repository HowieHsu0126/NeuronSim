"""Official Potjans--Diesmann microcircuit with independent seeds and pA controls.

The vendored network remains unchanged. Operational completion is separate from
single-cell and full-network scientific reference qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from .mechanistic import current_schedule

NATIVE_POPULATIONS = ('L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I')
SOURCE_FILES = ('network.py', 'network_params.py', 'sim_params.py',
                'stimulus_params.py', 'helpers.py', 'run_microcircuit.py', 'LICENSE')
CURRENT_READBACK_ATOL_PA = 1e-10
TIME_ALIGNMENT_ATOL_MS = 1e-9
EDGE_DTYPE = np.dtype([('target', '<i8'), ('source', '<i8'),
                       ('weight', '<f8'), ('delay', '<f8')])


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Check execution bounds and native-grid controls before creating a network."""
    cfg = deepcopy(config)
    if cfg.get('run_class') not in ('reference_candidate', 'engineering_diagnostic'):
        raise ValueError('run_class must distinguish reference_candidate and engineering_diagnostic')
    for key in ('duration_ms', 'dt_ms', 'n_scaling', 'k_scaling'):
        value = float(cfg[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        cfg[key] = value
    if max(cfg['n_scaling'], cfg['k_scaling']) > 1:
        raise ValueError('N and K scaling must not exceed the full source network')
    cfg['presimulation_ms'] = float(cfg['presimulation_ms'])
    if not 0 <= cfg['presimulation_ms'] < cfg['duration_ms']:
        raise ValueError('presimulation_ms must lie in [0, duration_ms)')
    for key in ('local_num_threads', 'fingerprint_target_batch_size'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['fingerprint_target_batch_size'] > 64:
        raise ValueError('fingerprint_target_batch_size must be at most 64 targets')
    if cfg['run_class'] == 'engineering_diagnostic':
        if (max(cfg['n_scaling'], cfg['k_scaling']) > .01 or
                cfg['duration_ms'] > 1000 or cfg['local_num_threads'] != 1):
            raise ValueError('diagnostic requires N/K <= 0.01, duration <= 1000 ms and one thread')
    elif cfg['local_num_threads'] < 4:
        raise ValueError('reference candidates require at least four threads')
    streams = cfg['random_streams']
    if set(streams) != {'connectivity', 'initial_state', 'background_drive'}:
        raise ValueError('three named random streams are required')
    for value in streams.values():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**32:
            raise ValueError('random stream seeds must be integers in [1, 2**32)')
    if cfg['initial_state'] not in ('original', 'optimized'):
        raise ValueError('initial_state must select an official source distribution')
    if not isinstance(cfg['poisson_input'], bool):
        raise ValueError('poisson_input must be boolean')
    cfg['background_rate_hz'] = float(cfg['background_rate_hz'])
    if not math.isfinite(cfg['background_rate_hz']) or cfg['background_rate_hz'] < 0:
        raise ValueError('background_rate_hz must be finite and nonnegative')
    edges, _ = pulse_schedule(cfg['current_pulses'], np.zeros(8), cfg['duration_ms'])
    for value in [*edges, cfg['presimulation_ms']]:
        if abs(round(value / cfg['dt_ms']) * cfg['dt_ms'] - value) > TIME_ALIGNMENT_ATOL_MS:
            raise ValueError('duration, presimulation and pulse edges must align with dt_ms')
    return cfg


def pulse_schedule(pulses: list[dict[str, Any]], baseline_pa: np.ndarray,
                   duration_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a common right-continuous grid of additive native-population currents."""
    baseline_pa = np.asarray(baseline_pa, dtype=float)
    if baseline_pa.shape != (len(NATIVE_POPULATIONS),) or not np.isfinite(baseline_pa).all():
        raise ValueError('one finite baseline current per native population is required')
    schedules = [current_schedule(pulses, pop, float(baseline_pa[i]), duration_ms,
                                  populations=NATIVE_POPULATIONS)
                 for i, pop in enumerate(NATIVE_POPULATIONS)]
    edges = np.unique(np.concatenate([times for times, _ in schedules]))
    values = np.column_stack([amplitudes[np.searchsorted(times, edges, side='right') - 1]
                              for times, amplitudes in schedules])
    return edges, values


def verify_source(source: Path) -> dict[str, Any]:
    provenance = json.loads((source / 'source.json').read_text())
    entries = {entry['path']: entry for entry in provenance['files']}
    if set(entries) != set(SOURCE_FILES) or provenance.get('git_tag') != 'v3.8':
        raise ValueError('expected the seven verified official v3.8 source files')
    for name in SOURCE_FILES:
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if digest != entries[name]['sha256']:
            raise ValueError(f'vendored official source changed: {name}')
    return provenance


def _load_module(source: Path, name: str):
    spec = importlib.util.spec_from_file_location(f'_neuronsim_potjans_{name}', source / f'{name}.py')
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load official source module {name}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_modules(source: Path):
    # The upstream network imports the unqualified name "helpers". Scope that
    # alias to loading the source, then restore the host interpreter's namespace.
    helper = _load_module(source, 'helpers')
    previous = sys.modules.get('helpers')
    sys.modules['helpers'] = helper
    try:
        network = _load_module(source, 'network')
    finally:
        if previous is None:
            del sys.modules['helpers']
        else:
            sys.modules['helpers'] = previous
    return (network, _load_module(source, 'network_params'),
            _load_module(source, 'sim_params'), _load_module(source, 'stimulus_params'))


def fingerprint_connections(nest, unit_ids: np.ndarray, batch_size: int) -> dict[str, Any]:
    """Hash neural incoming edges in canonical order using bounded target batches.

    All recurrent and background static-synapse inputs are included. Recorder
    links are excluded because their targets are not neural units. Memory is
    proportional to one small target batch's incoming edges, not the full graph.
    """
    ids = np.asarray(unit_ids, dtype=np.int64)
    if len(ids) == 0 or np.any(np.diff(ids) <= 0) or not 1 <= batch_size <= 64:
        raise ValueError('sorted unique unit IDs and a batch size in [1, 64] are required')
    digest = hashlib.sha256()
    total, largest = 0, 0
    delay_min, delay_max = float("inf"), -float("inf")
    for start in range(0, len(ids), batch_size):
        targets = nest.NodeCollection(ids[start:start + batch_size].tolist())
        connections = nest.GetConnections(target=targets, synapse_model='static_synapse')
        fields = connections.get(['source', 'target', 'weight', 'delay'])
        records = np.empty(len(fields['source']), dtype=EDGE_DTYPE)
        for name in records.dtype.names:
            records[name] = fields[name]
        if not np.isfinite(records['weight']).all() or not np.isfinite(records['delay']).all():
            raise ValueError('nonfinite native synapse parameters')
        if len(records):
            delay_min = min(delay_min, float(records['delay'].min()))
            delay_max = max(delay_max, float(records['delay'].max()))
        records.sort(order=['target', 'source', 'weight', 'delay'])
        digest.update(records.tobytes())
        total += len(records)
        largest = max(largest, len(records))
    return {'sha256': digest.hexdigest(), 'connection_count': total,
            'delay_min_ms': delay_min if total else None, 'delay_max_ms': delay_max if total else None,
            'target_batch_size': batch_size, 'largest_batch_connections': largest,
            'scope': 'static_synapse inputs to all neural units; recorder links excluded'}


def _set_rng_phase(nest, seed: int, phase: str) -> dict[str, Any]:
    if abs(float(nest.biological_time)) > TIME_ALIGNMENT_ATOL_MS:
        raise ValueError('RNG phase changes are only allowed before simulation')
    nest.rng_seed = seed
    actual = int(nest.rng_seed)
    if actual != seed:
        raise ValueError(f'native RNG seed readback failed during {phase}')
    return {'phase': phase, 'seed': actual, 'native_time_ms': float(nest.biological_time)}


def _read_currents(populations, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lower, upper, error = [], [], 0.
    for population, target in zip(populations, expected):
        actual = np.atleast_1d(np.asarray(population.get('I_e'), dtype=float))
        if len(actual) == 0 or not np.isfinite(actual).all():
            raise ValueError('native population current is missing or nonfinite')
        lower.append(float(actual.min()))
        upper.append(float(actual.max()))
        error = max(error, float(np.max(np.abs(actual - target))))
    if error > CURRENT_READBACK_ATOL_PA:
        raise ValueError(f'native population I_e differs from schedule by {error} pA')
    return np.array(lower), np.array(upper), error


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_reference(config: dict[str, Any], source: Path, output: Path) -> dict[str, Any]:
    """Run one native realization in this process; retain all neural unit IDs."""
    cfg = validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu_started = time.process_time()
    receipt: dict[str, Any] = {
        'status': 'RUNNING', 'scientific_qualification': 'NOT_ASSESSED',
        'resolved_config': cfg, 'source_directory': str(source),
        'start_unix': time.time(), 'command': sys.argv,
        'python': platform.python_version(), 'numpy': np.__version__,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'stage_costs_s': {},
        'initial_state_rng': 'numpy.PCG64', 'rng_phase_boundaries': [],
    }
    path = output / 'run.json'
    _write_receipt(path, receipt)
    try:
        # Native NEST is an optional runtime dependency; pure config/tests do
        # not import or initialize it.
        import nest

        receipt['nest_version'] = str(nest.__version__)
        if receipt['nest_version'] != str(cfg['expected_nest_version']):
            raise ValueError('native NEST version differs from the declared runtime')
        if int(nest.NumProcesses()) != 1:
            raise ValueError('this artifact writer supports one MPI rank with multiple threads')
        receipt['source'] = verify_source(source)
        phase_start = time.perf_counter()
        network_module, network_params, sim_params, stimulus_params = load_source_modules(source)
        net_dict, sim_dict, stim_dict = (deepcopy(network_params.net_dict),
                                       deepcopy(sim_params.sim_dict), deepcopy(stimulus_params.stim_dict))
        if tuple(net_dict['populations']) != NATIVE_POPULATIONS:
            raise ValueError('official source population order changed')
        net_dict.update(N_scaling=cfg['n_scaling'], K_scaling=cfg['k_scaling'],
                        V0_type=cfg['initial_state'], poisson_input=cfg['poisson_input'],
                        bg_rate=cfg['background_rate_hz'])
        if np.any(np.round(net_dict['full_num_neurons'] * cfg['n_scaling']) < 1):
            raise ValueError('the selected scale removes a native population')
        sim_dict.update(data_path=str(output / 'source_data'), sim_resolution=cfg['dt_ms'],
                        rng_seed=cfg['random_streams']['connectivity'],
                        local_num_threads=cfg['local_num_threads'], overwrite_files=False,
                        print_time=False, rec_dev=['spike_recorder'])
        # Adapter controls use actual per-population pA, not the upstream
        # dc_amp*K_ext convenience control. Preserve the source's baseline I_e.
        stim_dict.update(dc_input=False, thalamic_input=False)
        nest.set_verbosity('M_WARNING')
        network = network_module.Network(sim_dict, net_dict, stim_dict)
        network.create()
        network.spike_recorders.set(record_to='memory')
        rng = np.random.Generator(np.random.PCG64(cfg['random_streams']['initial_state']))
        ids, membership, initial = [], [], []
        baseline = []
        for index, (label, population) in enumerate(zip(NATIVE_POPULATIONS, network.pops)):
            params = net_dict['neuron_params']
            mean = params['V0_mean'][cfg['initial_state']]
            std = params['V0_std'][cfg['initial_state']]
            if cfg['initial_state'] == 'optimized':
                mean, std = mean[index], std[index]
            vm = rng.normal(float(mean), float(std), len(population))
            population.set(V_m=vm, C_m=float(params['C_m']), tau_m=float(params['tau_m']))
            population_ids = np.asarray(population.tolist(), dtype=np.int64)
            ids.extend(population_ids)
            membership.extend([label] * len(population))
            initial.extend(vm)
            actual = np.atleast_1d(np.asarray(population.get('I_e'), dtype=float))
            if not np.isfinite(actual).all() or not np.all(actual == float(network.DC_amp[index])):
                raise ValueError('native baseline current differs from official network compensation')
            baseline.append(float(actual[0]))
        unit_ids = np.asarray(ids, dtype=np.int64)
        population_labels = np.asarray(membership)
        initial_vm = np.asarray(initial, dtype=np.float64)
        if np.any(np.diff(unit_ids) <= 0):
            raise ValueError('native neural unit IDs must be ordered and unique')
        receipt['stage_costs_s']['source_and_nodes'] = time.perf_counter() - phase_start
        receipt['rng_phase_boundaries'].append(_set_rng_phase(
            nest, cfg['random_streams']['connectivity'], 'connectivity'))
        phase_start = time.perf_counter()
        network.connect()
        receipt['stage_costs_s']['connect_and_prepare'] = time.perf_counter() - phase_start
        actual_vm = np.concatenate([np.atleast_1d(pop.get('V_m')) for pop in network.pops])
        if not np.array_equal(actual_vm, initial_vm):
            raise ValueError('native preparation changed the independently supplied initial voltage')
        receipt['initial_state_fingerprint_sha256'] = hashlib.sha256(
            unit_ids.astype('<i8').tobytes() + initial_vm.astype('<f8').tobytes()).hexdigest()
        phase_start = time.perf_counter()
        fingerprint = fingerprint_connections(nest, unit_ids, cfg['fingerprint_target_batch_size'])
        receipt['connectivity'] = fingerprint
        receipt['connectivity_fingerprint_sha256'] = fingerprint['sha256']
        receipt['connection_count'] = fingerprint['connection_count']
        receipt['stage_costs_s']['connectivity_fingerprint'] = time.perf_counter() - phase_start
        receipt['rng_phase_boundaries'].append(_set_rng_phase(
            nest, cfg['random_streams']['background_drive'], 'background_drive'))
        receipt['native_kernel'] = {'local_num_threads': int(nest.local_num_threads),
                                    'total_num_virtual_procs': int(nest.total_num_virtual_procs),
                                    'resolution_ms': float(nest.resolution),
                                    'num_processes': int(nest.NumProcesses())}
        edges, expected = pulse_schedule(cfg['current_pulses'], np.array(baseline), cfg['duration_ms'])
        ticks = np.rint(edges / cfg['dt_ms']).astype(np.int64)
        lower, upper, interval_lower, interval_upper = [], [], [], []
        delivery_error = 0.
        phase_start = time.perf_counter()
        for index, tick in enumerate(ticks):
            if abs(float(nest.biological_time) - tick * cfg['dt_ms']) > TIME_ALIGNMENT_ATOL_MS:
                raise ValueError('native simulation time differs from the requested current edge')
            for pop_index, population in enumerate(network.pops):
                population.set(I_e=float(expected[index, pop_index]))
            lo, hi, error = _read_currents(network.pops, expected[index])
            lower.append(lo)
            upper.append(hi)
            delivery_error = max(delivery_error, error)
            if index + 1 < len(ticks):
                nest.Simulate(float((ticks[index + 1] - tick) * cfg['dt_ms']))
                lo, hi, error = _read_currents(network.pops, expected[index])
                interval_lower.append(lo)
                interval_upper.append(hi)
                delivery_error = max(delivery_error, error)
        receipt['stage_costs_s']['simulation_and_control_readback'] = time.perf_counter() - phase_start
        spike_times, spike_ids = [], []
        for recorder in network.spike_recorders:
            events = recorder.get('events')
            spike_times.append(np.asarray(events['times'], dtype=np.float64) / 1000.)
            spike_ids.append(np.asarray(events['senders'], dtype=np.int64))
        times_s = np.concatenate(spike_times)
        senders = np.concatenate(spike_ids)
        if (not np.isfinite(times_s).all() or
                np.any((times_s < 0) | (times_s > cfg['duration_ms'] / 1000. + 1e-12)) or
                not np.isin(senders, unit_ids).all()):
            raise ValueError('invalid native spike identities or times')
        order = np.lexsort((senders, times_s))
        times_s, senders = times_s[order], senders[order]
        phase_start = time.perf_counter()
        np.savez_compressed(output / 'spikes.npz', spike_times_s=times_s, spike_unit_ids=senders,
                            unit_ids=unit_ids, population=population_labels, initial_voltage_mv=initial_vm)
        np.savez_compressed(output / 'delivered_currents.npz', time_ms=edges,
                            populations=np.asarray(NATIVE_POPULATIONS), baseline_pa=np.array(baseline),
                            expected_pa=expected, actual_min_pa=np.array(lower), actual_max_pa=np.array(upper),
                            interval_end_min_pa=np.array(interval_lower), interval_end_max_pa=np.array(interval_upper))
        analysis_mask = times_s >= cfg['presimulation_ms'] / 1000.
        analysis_seconds = (cfg['duration_ms'] - cfg['presimulation_ms']) / 1000.
        receipt.update(
            status='COMPLETED', neural_units=len(unit_ids), spikes=len(senders),
            delivery_max_error_pa=delivery_error,
            population_unit_counts={label: int(np.sum(population_labels == label)) for label in NATIVE_POPULATIONS},
            analysis_interval_ms=[cfg['presimulation_ms'], cfg['duration_ms']],
            population_rates_hz={label: float(np.isin(senders[analysis_mask], unit_ids[population_labels == label]).sum()
                                             / np.sum(population_labels == label) / analysis_seconds)
                                 for label in NATIVE_POPULATIONS},
            spike_fingerprint_sha256=hashlib.sha256(senders.astype('<i8').tobytes() + times_s.astype('<f8').tobytes()).hexdigest(),
            artifacts={'spikes': 'spikes.npz', 'current_delivery': 'delivered_currents.npz'},
        )
        receipt['stage_costs_s']['serialization'] = time.perf_counter() - phase_start
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error))
        (output / 'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        receipt['end_unix'] = time.time()
        receipt['wall_time_s'] = time.perf_counter() - started
        receipt['cpu_time_s'] = time.process_time() - cpu_started
        receipt['peak_process_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        _write_receipt(path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--n-scaling', type=float)
    parser.add_argument('--k-scaling', type=float)
    parser.add_argument('--duration-ms', type=float)
    parser.add_argument('--threads', type=int)
    for name in ('connectivity', 'initial-state', 'background-drive'):
        parser.add_argument(f'--{name}-seed', type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError('reference config must be a mapping')
    for option, key in (('n_scaling', 'n_scaling'), ('k_scaling', 'k_scaling'),
                        ('duration_ms', 'duration_ms'), ('threads', 'local_num_threads')):
        value = getattr(args, option)
        if value is not None:
            config[key] = value
    for stream in ('connectivity', 'initial_state', 'background_drive'):
        value = getattr(args, stream + '_seed')
        if value is not None:
            config['random_streams'][stream] = value
    config['run_class'] = 'engineering_diagnostic' if args.diagnostic else 'reference_candidate'
    config['config_path'] = str(config_path)
    source = Path(config['source_dir'])
    if not source.is_absolute():
        source = config_path.parents[2] / source
    receipt = run_reference(config, source.resolve(), args.output.resolve())
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()

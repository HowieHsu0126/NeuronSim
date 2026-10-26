"""Replay the ModelDB 266872 network with explicit random streams and currents.

Runs use a node-local build directory. Population-current stimulation is a
simulation control, not a claim about arbitrary-neuron optogenetic access.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

POPULATIONS = ('L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i')
SOURCE_FILES = ('cfg.py', 'netParams.py', 'IntFire_PD.mod', 'LICENSE', 'source.json')


def current_schedule(pulses: list[dict[str, Any]], population: str,
                     baseline_pa: float, duration_ms: float,
                     populations: tuple[str, ...] = POPULATIONS) -> tuple[np.ndarray, np.ndarray]:
    """Right-continuous, additive population currents; all times are in ms."""
    if not np.isfinite(duration_ms) or duration_ms <= 0 or not np.isfinite(baseline_pa):
        raise ValueError('finite positive duration and finite baseline required')
    relevant = []
    for pulse in pulses:
        if pulse['population'] not in populations:
            raise ValueError('unknown current target population')
        start, width, amplitude = (float(pulse[k]) for k in ('onset_ms', 'duration_ms', 'amplitude_pa'))
        if not np.isfinite([start, width, amplitude]).all() or start < 0 or width <= 0 or start + width > duration_ms:
            raise ValueError('current pulse must lie inside the recorded simulation')
        if pulse['population'] == population:
            relevant.append((start, start + width, amplitude))
    times = np.unique([0., duration_ms] + [edge for start, end, _ in relevant for edge in (start, end)])
    values = np.full(times.shape, baseline_pa, dtype=float)
    for start, end, amplitude in relevant:
        values[(times >= start) & (times < end)] += amplitude
    return times, values


def prepare_reference(source: Path, build: Path, output: Path) -> dict[str, Any]:
    build.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, build / name)
    provenance = json.loads((source / 'source.json').read_text())
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='1')
    builder = Path(sys.executable).parent / 'nrnivmodl'
    if not builder.is_file():
        raise FileNotFoundError('nrnivmodl is required in the reference Python environment')
    with (output / 'build.log').open('w') as log:
        subprocess.run([str(builder)], cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return provenance


def run_reference(config: dict[str, Any], source: Path, build: Path, output: Path) -> dict[str, Any]:
    """Execute one network realization in a fresh process and save spike events."""
    if config.get('run_class') != 'engineering_diagnostic':
        raise ValueError(
            'ModelDB 266872 has a verified refractory PSC semantic mismatch; '
            'only --diagnostic runs are supported. See synaptic_reference.py.'
        )
    duration = float(config['duration_ms'])
    dt = float(config['dt_ms'])
    scale = float(config['network_scale'])
    record_dt = float(config['current_record_dt_ms'])
    if not np.isfinite([duration, dt, scale, record_dt]).all() or duration <= 0 or dt <= 0 or record_dt < dt or not 0 < scale <= 1:
        raise ValueError('invalid reference time, scale or recording settings')
    pulses = config['current_pulses']
    for pop in POPULATIONS:
        current_schedule(pulses, pop, 0., duration)
    streams = config['random_streams']
    if set(streams) != {'connectivity', 'initial_state', 'background_drive'}:
        raise ValueError('three named mechanistic random streams are required')
    for value in streams.values():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError('reference random stream seeds must be uint32 integers')
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {'status': 'RUNNING', 'scientific_qualification': 'NOT_ASSESSED',
        'resolved_config': config, 'start_unix': time.time(), 'command': sys.argv,
        'python': platform.python_version(), 'numpy': np.__version__,
        'cpu_affinity': sorted(os.sched_getaffinity(0))}
    receipt_path = output / 'run.json'
    try:
        receipt['source'] = prepare_reference(source, build, output)
        # NEURON/NetPyNE load compiled mechanisms and belong only in this command.
        os.chdir(build)
        sys.path.insert(0, str(build))
        from netpyne import sim
        from neuron import h
        cfg = importlib.import_module('cfg').cfg
        cfg.ScaleFactor, cfg.duration, cfg.dt = scale, duration, dt
        cfg.Balanced, cfg.DC, cfg.TH = bool(config['balanced_background']), bool(config['dc_background']), False
        cfg.seeds.update(conn=streams['connectivity'], stim=streams['background_drive'], m=streams['initial_state'])
        cfg.savePickle = cfg.saveJson = cfg.saveCellConns = cfg.saveCellSecs = False
        cfg.printPopAvgRates = cfg.printRunTime = False
        net = importlib.import_module('netParams').netParams
        # The artificial-cell update clock must change together with dt.
        net.popParams['bkg_IF']['rate'] = 1000. / dt
        cfg.analysis = {}
        sim.initialize(simConfig=cfg, netParams=net)
        sim.net.createPops()
        sim.net.createCells()
        rand = h.Random()
        ids, memberships, playback, recordings, schedules = [], [], [], {}, {}
        for cell in sim.net.cells:
            if cell.tags['cellModel'] != 'IntFire_PD':
                continue
            rand.Random123(cell.gid, cfg.seeds['m'])
            cell.hPointp.m = rand.normal(-58, 10)  # Source normal uses variance 10.
            ids.append(cell.gid)
            pop = cell.tags['pop']
            memberships.append(pop)
            if pop not in schedules:
                schedules[pop] = current_schedule(pulses, pop, float(cell.hPointp.Iext), duration)
                recordings[pop] = h.Vector().record(cell.hPointp._ref_Iext, record_dt)
            times, values = schedules[pop]
            tv, vv = h.Vector(times), h.Vector(values)
            vv.play(cell.hPointp._ref_Iext, tv, 0)
            playback.extend((tv, vv))  # Keep every Vector alive until run completion.
        clock = h.Vector().record(h._ref_t, record_dt)
        sim.net.addStims()
        sim.net.connectCells()
        connection_count = sum(len(c.conns) for c in sim.net.cells)
        topology = hashlib.sha256()
        for cell in sorted(sim.net.cells, key=lambda c: c.gid):
            entries = sorted((int(c['preGid']), float(c['weight']), float(c['delay'])) for c in cell.conns)
            for pre, weight, delay in entries:
                topology.update(struct.pack('<qqdd', int(cell.gid), pre, weight, delay))
        receipt['connectivity_fingerprint_sha256'] = topology.hexdigest()
        sim.setupRecording()
        sim.runSim()
        sim.gatherData()
        times_s = np.asarray(sim.allSimData['spkt'], dtype=float) / 1000.
        spike_ids = np.asarray(sim.allSimData['spkid'], dtype=np.int64)
        ids_array, labels = np.asarray(ids, dtype=np.int64), np.asarray(memberships)
        if not np.isfinite(times_s).all() or ((times_s < 0) | (times_s > duration / 1000.)).any() or not np.isin(spike_ids, ids_array).all():
            raise ValueError('invalid reference spike identity or time')
        tt = np.asarray(clock)
        delivered = {}
        max_error = 0.
        for pop, vector in recordings.items():
            values = np.asarray(vector)
            edges, amplitudes = schedules[pop]
            expected = amplitudes[np.searchsorted(edges, tt, side='right') - 1]
            interior = np.ones(tt.shape, dtype=bool)
            for edge in edges:
                interior &= np.abs(tt - edge) > record_dt + 1e-9
            error = float(np.max(np.abs(values[interior] - expected[interior]), initial=0.))
            max_error = max(max_error, error)
            delivered[pop] = values
        if max_error > 1e-8:
            raise ValueError('current playback disagrees with the declared pulse schedule')
        np.savez_compressed(output / 'spikes.npz', spike_times_s=times_s, spike_unit_ids=spike_ids,
                            unit_ids=ids_array, population=labels)
        np.savez_compressed(output / 'delivered_currents.npz', time_ms=tt, **delivered)
        receipt.update(status='COMPLETED', neural_units=len(ids), spikes=len(times_s),
                       connection_count=connection_count, update_clock_hz=1000. / dt,
                       delivery_max_error_away_from_event_boundaries_pa=max_error,
                       neuron_version=h.nrnversion(0),
                       population_rates_hz={pop: float(np.isin(spike_ids, ids_array[labels == pop]).sum() /
                           (labels == pop).sum() / (duration / 1000.)) for pop in POPULATIONS})
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error))
        raise
    finally:
        receipt['end_unix'] = time.time()
        receipt['wall_time_s'] = receipt['end_unix'] - receipt['start_unix']
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network-scale', type=float)
    parser.add_argument('--background-drive-seed', type=int)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError('reference configuration must be a mapping')
    if args.network_scale is not None:
        config['network_scale'] = args.network_scale
    if args.background_drive_seed is not None:
        config['random_streams']['background_drive'] = args.background_drive_seed
    config['run_class'] = 'engineering_diagnostic' if args.diagnostic else 'reference_candidate'
    source = Path(config['source_dir'])
    if not source.is_absolute():
        source = config_path.parents[2] / source
    print(json.dumps(run_reference(config, source.resolve(), args.build_dir.resolve(), args.output.resolve()), allow_nan=False))


if __name__ == '__main__':
    main()

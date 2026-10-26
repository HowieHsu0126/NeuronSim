"""Conditional focal first-passage Monte Carlo against a validated PFC reference.

Each sample changes only the focal future innovations.  A second spike absorbs
the trajectory; later feedback is outside this calculation.  Monte Carlo
coverage assumes nominal IID innovations and is not population-history coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import binomtest
import yaml

from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native


def require(condition, message):
    if not condition:
        raise ValueError('PFC_FIRST_PASSAGE_PROBABILITY_' + message)


def first_passage_ticks(config, conditional, forcing, counts):
    """Apply X_tau..X_(b-2), then test source thresholds tau+1..b-1.

    ``counts`` has sample-major shape [samples, input_slots].  Observed future
    focal voltage and AMPA are deliberately disallowed as forcing inputs.
    """
    require(conditional['population'] == 'E' and conditional['fresh_input_included'] is False
            and conditional['timing'] == 'first_spike_reset_with_preinput_ampa', 'CONDITIONAL_INITIAL_TIMING')
    require('s_AMPA' not in forcing and 'v' not in forcing, 'FOCAL_FUTURE_STATE_LEAKAGE')
    tau, stop = conditional['first_spike_tick'], conditional['window_stop_tick']
    dt = config['dt_ms'] * 1e-3
    ticks = np.asarray(forcing['tick'])
    require(ticks.dtype.kind in 'iu' and np.array_equal(ticks, np.arange(tau + 1, stop)), 'FORCING_TICKS')
    require(np.array_equal(forcing['time_seconds'], ticks * dt), 'FORCING_TIME_UNITS')
    required = ('s_GABA', 's_NMDA_total', 'I_stim', 's_NMDA', 'x')
    for name in required:
        value = np.asarray(forcing[name])
        require(value.shape == ticks.shape and value.dtype == np.float64 and np.isfinite(value).all(), 'FORCING_SCHEMA:' + name)
    counts = np.asarray(counts)
    require(counts.dtype == np.uint16 and counts.ndim == 2 and counts.shape[0] > 0
            and counts.shape[1] == len(ticks) and np.all(counts <= 1000), 'COMPLETE_SOURCE_INPUTS')
    constants = source._constants(config, 'E')
    initial = conditional['state']
    require(initial['v'] == constants['reset'] and initial['lastspike'] == tau * dt
            and initial['not_refractory'] is False, 'FIRST_SPIKE_RESET')
    # These deterministic reference coordinates are common to every sample.
    # NumPy broadcasts their one-element arrays through the unchanged source
    # RK2 expressions. Focal voltage, AMPA and refractory state stay per sample.
    shared = {'s_GABA', 's_NMDA_total', 'I_stim', 's_NMDA', 'x'}
    state = {key: np.full(1 if key in shared else counts.shape[0], value,
                          dtype=bool if key == 'not_refractory' else np.float64)
             for key, value in initial.items()}
    require(all(np.isfinite(value).all() for value in state.values()), 'NONFINITE_INITIAL_STATE')
    arithmetic = source._Arithmetic('numpy')
    alive = np.ones(counts.shape[0], dtype=bool)
    first = np.full(counts.shape[0], -1, dtype=np.int64)
    for j, tick in enumerate(ticks):
        # The unknown fresh row is added after tau's reset and before tau+1 RK2.
        state['s_AMPA'] += counts[:, j]
        for key in ('s_GABA', 's_NMDA_total', 'I_stim'):
            state[key].fill(forcing[key][j])
        # NMDA state does not depend on subthreshold focal voltage or AMPA.
        require(not alive.any() or (state['s_NMDA'][0] == forcing['s_NMDA'][j]
                and state['x'][0] == forcing['x'][j]), 'REFERENCE_NMDA_STATE')
        # Absorbed samples no longer represent source paths. Keep them inert so
        # unused post-hit arithmetic cannot overflow or affect later checks.
        state['v'][~alive] = constants['reset']
        state['s_AMPA'][~alive] = 0.0
        state['lastspike'][~alive] = tick * dt
        source._rk2(state, constants, tick * dt, dt, arithmetic, 'E')
        require(np.isfinite(state['v'][alive]).all(), 'NONFINITE_ACTIVE_TRANSITION')
        hit = alive & state['not_refractory'] & (state['v'] > constants['threshold'])
        first[hit] = tick
        alive[hit] = False
    return first

def binomial_interval(hits, trials, confidence):
    require(type(hits) is int and type(trials) is int and 0 <= hits <= trials and trials > 0,
            'BERNOULLI_DENOMINATOR')
    require(0. < confidence < 1., 'CONFIDENCE_LEVEL')
    interval = binomtest(hits, trials).proportion_ci(confidence_level=confidence, method='exact')
    return dict(lower=float(interval.low), upper=float(interval.high), confidence_level=confidence,
                method='two_sided_Clopper_Pearson', scope='conditional_auxiliary_Bernoulli_trials_under_nominal_IID_input_law')


def monte_carlo(config, conditional, forcing, *, samples, batch_size, seed_components,
                confidence=.95, progress: Callable | None = None, failure_output: Path | None = None):
    require(type(samples) is int and samples > 0 and type(batch_size) is int and batch_size > 0,
            'SAMPLING_BUDGET')
    seeds = np.asarray(seed_components)
    require(seeds.ndim == 1 and seeds.size > 0 and seeds.dtype.kind in 'iu'
            and np.all((seeds >= 0) & (seeds < 2**32)), 'AUXILIARY_SEED_ARRAY')
    rng = np.random.RandomState(seeds.astype(np.uint32))
    probability = native._RATE_HZ * (config['dt_ms'] * 1e-3)
    slots = len(forcing['tick'])
    histogram = np.zeros(slots, dtype=np.int64)
    completed = hits = 0
    input_digest = hashlib.sha256()
    started = time.perf_counter()
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        # One persistent stream; sample-major layout preserves the declared
        # sequence of focal future rows. Never reseed an individual batch.
        draws = rng.binomial(native._BINOMIAL_N, probability, size=(count, slots)).astype(np.uint16)
        input_digest.update(memoryview(draws))
        try:
            first = first_passage_ticks(config, conditional, forcing, draws)
        except Exception:
            if failure_output is not None:
                np.savez_compressed(failure_output/('failed_batch_'+str(start)+'.npz'),
                                    counts=draws, batch_start=np.array(start))
            raise
        observed = first[first >= 0]
        histogram += np.bincount(observed - int(forcing['tick'][0]), minlength=slots)
        completed += count; hits += len(observed)
        if progress is not None:
            progress(dict(completed=completed, hits=hits, wall_seconds=time.perf_counter()-started))
    require(completed == samples and int(histogram.sum()) == hits, 'HIT_DENOMINATOR')
    state = rng.get_state()
    final_rng = dict(algorithm=state[0], key=state[1].tolist(), position=int(state[2]),
                     has_gauss=int(state[3]), cached_gaussian=float(state[4]))
    numerator, denominator = probability.as_integer_ratio()
    result = dict(status='COMPLETE_CONDITIONAL_FIRST_PASSAGE_MONTE_CARLO', samples=samples, hits=hits,
        no_hits=samples-hits, estimate=hits/samples, interval=binomial_interval(int(hits), samples, confidence),
        input_slots=slots, binomial_n=native._BINOMIAL_N, probability=probability,
        probability_numerator=numerator, probability_denominator=denominator,
        seed_components=list(map(int, seeds)), batch_size=batch_size, draw_layout='sample_major_complete_focal_future',
        input_draw_sha256=input_digest.hexdigest(), final_rng_state=final_rng, numpy_version=np.__version__,
        scipy_version=scipy_version, wall_seconds=time.perf_counter()-started,
        source_kernel='unchanged_numpy_RK2', first_hit_histogram_is_simultaneous_band=False,
        finite_PRNG_law_certified=False, population_history_uncertainty_established=False,
        population_first_passage_probability_established=False, third_spike_probability_established=False,
        new_full_network_episodes=0, new_independent_source_blocks=0,
        confirmation_execution_released=False, scientific_fidelity_qualified=False, production_episodes=0)
    return result, histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    config = yaml.safe_load(args.source_config.read_text())
    conditional = json.loads((args.reference/'conditional_initial.json').read_text())
    with np.load(args.reference/'reference_window.npz', allow_pickle=False) as data:
        forcing = {key:data[key] for key in data.files}
    require(settings['samples'] == 1000000 and settings['batch_size'] == 10000
            and settings['seed_components'] == [20261009,95001,28022,2004]
            and settings['confidence_level'] == .95, 'FROZEN_DEVELOPMENT_SAMPLING_BUDGET')
    require(conditional['unit_index'] == 28022 and conditional['first_spike_tick'] == 2004
            and conditional['window_stop_tick'] == 2100 and len(forcing['tick']) == 95
            and config['dt_ms'] == .05, 'REGISTERED_REFERENCE_WINDOW')
    args.output.mkdir(parents=True, exist_ok=False)
    def progress(row):
        print(json.dumps(row), flush=True)
        (args.output/'sampling_progress.json').write_text(json.dumps(row, indent=2)+'\n')
    result, histogram = monte_carlo(config, conditional, forcing, samples=settings['samples'],
        batch_size=settings['batch_size'], seed_components=settings['seed_components'],
        confidence=settings['confidence_level'], progress=progress, failure_output=args.output)
    result['conditioning'] = 'first_spike_past_and_other_unit_future_innovations; observed_no_hit_is_not_an_added_condition'
    result['reference_archive'] = settings['reference']['archive']
    result['reference_delivery_sha256'] = settings['reference']['delivery_sha256']
    np.savez_compressed(args.output/'first_hit_histogram.npz', tick=forcing['tick'], hits=histogram)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','samples','hits','estimate','interval')}), flush=True)


if __name__ == '__main__':
    main()

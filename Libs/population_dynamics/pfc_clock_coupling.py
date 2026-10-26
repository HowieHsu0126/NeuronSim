"""Binomial clock refinement with unchanged coarse PFC source randomness.

This is an external-input coupling candidate, not a source-science release.
For each parent Bernoulli(p) success, assign zero or two children with
probability p/4 each, otherwise one uniformly chosen child. Parent failures
have no children. The unconditional child pair is two independent
Bernoulli(p/2) variables. Aggregating N independent trials gives independent
Binomial(N,p/2) child counts. Applying this kernel independently to both
children gives the next clock level without changing either marginal law.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
import math
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from .pfc_native_drive import create_drive, _BINOMIAL_N, _RATE_HZ, _seed, _state_arrays
from .provenance import freeze_json, sha256_file

CLOCK_LABELS = ('coarse', 'middle', 'fine')


# NumPy 2.4.2 RandomState's legacy inversion uses exp(n*log(1-p)) and
# sequential residual subtraction. At the PFC clocks most parent counts are
# zero. Reuse the small-count coefficients while drawing every native uniform,
# including zero-count positions. Other kernels retain native dispatch.
_SMALL_COUNT_LIMIT = 8


@lru_cache(maxsize=16)
def _small_binomial_masses(probability):
    q = 1.0 - probability
    masses = np.zeros((_SMALL_COUNT_LIMIT + 1, _SMALL_COUNT_LIMIT + 1))
    for n in range(_SMALL_COUNT_LIMIT + 1):
        masses[n, 0] = math.exp(n * math.log(q))
        for k in range(1, n + 1):
            masses[n, k] = ((n - k + 1) * probability * masses[n, k - 1]) / (k * q)
    masses.flags.writeable = False
    return masses


def _small_binomial_from_uniforms(counts, uniforms, masses):
    """Use native residual comparisons; None requests the native restart path."""
    result = np.zeros(counts.shape, dtype=np.int64)
    active = np.flatnonzero(uniforms > masses[counts, 0])
    k = 0
    while active.size:
        k += 1
        if np.any(k > counts[active]):
            # Native inversion starts another scalar draw beyond its bound.
            # Restore the whole stream rather than reorder those extra draws.
            return None
        uniforms[active] -= masses[counts[active], k - 1]
        result[active] = k
        active = active[uniforms[active] > masses[counts[active], k]]
    return result


def _refinement_binomial(counts, probability, rng):
    """Preserve seeded native draws in the qualified small-count domain."""
    if (type(rng) is not np.random.RandomState or np.__version__ != '2.4.2'
            or not (0.0 < probability <= 1e-4 or probability == 0.5)
            or int(counts.max()) > _SMALL_COUNT_LIMIT):
        return rng.binomial(counts, probability)
    before = rng.get_state()
    result = _small_binomial_from_uniforms(
        counts, rng.random_sample(counts.size), _small_binomial_masses(probability))
    if result is None:
        rng.set_state(before)
        return rng.binomial(counts, probability)
    return result


def split_counts(parent, probability, rng):
    """Return consecutive half-tick counts, conditional on a parent count.

    The two-stage binomial draw is a multinomial over parent successes with
    probabilities (p/4, p/4, 1-p/2) for zero, double and single children.
    Fresh auxiliary randomness must be independent of the coarse stream.
    """
    x = np.asarray(parent)
    if (x.ndim != 1 or x.size == 0 or x.dtype.kind not in 'iu'
            or np.any(x > _BINOMIAL_N) or np.any(x < 0)
            or isinstance(probability, (bool, np.bool_))
            or not np.isfinite(probability) or not 0 < probability <= 1):
        raise ValueError('VALID_PARENT_BINOMIAL_COUNTS_AND_PROBABILITY_REQUIRED')
    x = x.astype(np.int64, copy=False)
    zero_probability = probability / 4.0
    zeros = _refinement_binomial(x, zero_probability, rng)
    doubles = _refinement_binomial(x - zeros, zero_probability / (1.0 - zero_probability), rng)
    singles = x - zeros - doubles
    left_singles = _refinement_binomial(singles, 0.5, rng)
    return np.stack((doubles + left_singles, doubles + singles - left_singles)).astype(np.uint16)


class CoupledClockRows:
    """One native coarse row plus two/four coupled finer rows in I/E order.

    Generation order is one complete coarse tick at a time, independent of
    a future consumer's block size. No full input trajectory is retained.
    """

    def __init__(self, coarse_drive, refinement_seeds):
        if (coarse_drive.ticks != 0 or len(refinement_seeds) != 2
                or len(set(refinement_seeds)) != 2):
            raise ValueError('FRESH_COARSE_DRIVE_AND_TWO_DISTINCT_REFINEMENT_SEEDS_REQUIRED')
        seeds = [_seed(value) for value in refinement_seeds]
        if set(seeds) & set(coarse_drive.seeds.values()):
            raise ValueError('REFINEMENT_STREAMS_MUST_NOT_REUSE_SOURCE_SEEDS')
        self.coarse = coarse_drive
        self.refinement_seeds = seeds
        self.children = [np.random.RandomState(seed) for seed in seeds]

    def take_row(self):
        native = self.coarse.take(1)
        packed = np.concatenate((native['I'][0], native['E'][0]))
        middle = split_counts(packed, self.coarse.probability, self.children[0])
        fine = np.concatenate([split_counts(row, self.coarse.probability / 2, self.children[1])
                               for row in middle], axis=0)
        return packed[None, :], middle, fine

    def state_arrays(self):
        states = {'coarse': self.coarse.get_state()}
        states.update({f'refinement_{level+1}': rng.get_state() for level, rng in enumerate(self.children)})
        return {name + '__' + key: value for name, state in states.items()
                for key, value in _state_arrays(state).items()}

    def metadata(self):
        return dict(coupling='binomial_parent_refinement', coarse_source=self.coarse.metadata(),
                    refinement_seeds=self.refinement_seeds,
                    auxiliary_rng='numpy.random.RandomState(MT19937)',
                    auxiliary_draw_order='coarse_tick_then_level_then_subtick; packed_I_then_E',
                    fine_source_seed_sequence_preserved=False,
                    fine_binomial_marginals_preserved_by_construction=True,
                    source_simulations=0, neural_error_reduction_established=False)


def _moments(histogram, offset=0):
    support = np.arange(len(histogram), dtype=np.float64) + offset
    count = int(histogram.sum())
    mean = float(support @ histogram / count)
    variance = float(((support - mean)**2) @ histogram / count)
    return dict(count=count, mean=mean, variance=variance)


def run(config_path, project_root, output):
    settings = yaml.safe_load(config_path.read_text())
    if (settings.get('schema') != 'pfc_clock_coupling_diagnostic_v1'
            or settings.get('scope') != 'development_driver_diagnostic'
            or settings.get('confirmation_execution_released') is not False):
        raise ValueError('DEVELOPMENT_INPUT_COUPLING_SCOPE_REQUIRED')
    source_path = project_root / settings['source_config']
    source = yaml.safe_load(source_path.read_text())
    if (source['excitatory_neurons'] != 80000 or source['inhibitory_neurons'] != 20000
            or source['dt_ms'] != 0.05 or settings['coarse_ticks'] != 1000):
        raise ValueError('BOUNDED_100K_SOURCE_INPUT_DIAGNOSTIC_REQUIRED')
    config = dict(source, duration_ms=settings['coarse_ticks'] * source['dt_ms'])
    case = dict(case_id='pfc_clock_coupling_input_diagnostic', source_rng_seed=settings['source_rng_seed'])
    initial, coarse = create_drive(config, case)
    drive = CoupledClockRows(coarse, settings['refinement_seeds'])
    output.mkdir(parents=True, exist_ok=False)
    histograms = {label: np.zeros(_BINOMIAL_N + 1, dtype=np.int64) for label in CLOCK_LABELS}
    # Signed sum differences can span [-N,3N]. Keep the full mathematical domain.
    differences = {label: np.zeros(4 * _BINOMIAL_N + 1, dtype=np.int64)
                   for label in ('middle_minus_coarse', 'fine_minus_coarse', 'fine_minus_middle')}
    started = time.perf_counter()
    for tick in range(settings['coarse_ticks']):
        rows = drive.take_row()
        for label, values in zip(CLOCK_LABELS, rows):
            histograms[label] += np.bincount(values.ravel(), minlength=_BINOMIAL_N + 1)
        coarse_counts = rows[0][0].astype(np.int64)
        delta = {
            'middle_minus_coarse': rows[1].sum(axis=0, dtype=np.int64) - coarse_counts,
            'fine_minus_coarse': rows[2].sum(axis=0, dtype=np.int64) - coarse_counts,
            'fine_minus_middle': (rows[2].reshape(2, 2, -1).sum(axis=1, dtype=np.int64)
                                  - rows[1].astype(np.int64)).ravel(),
        }
        for label, values in delta.items():
            differences[label] += np.bincount(values + _BINOMIAL_N, minlength=4 * _BINOMIAL_N + 1)
        if (tick + 1) % 100 == 0:
            print(json.dumps(dict(coarse_ticks=tick+1)), flush=True)
    elapsed = time.perf_counter() - started
    counts_path = output / 'input_count_histograms.npz'
    np.savez(counts_path, **histograms, **differences)
    rng_path = output / 'final_rng.npz'
    np.savez(rng_path, **drive.state_arrays())
    probability = coarse.probability
    marginals = {}
    for level, label in enumerate(CLOCK_LABELS):
        p = probability / (2**level)
        marginals[label] = dict(_moments(histograms[label]), dt_ms=source['dt_ms']/(2**level),
                                binomial_n=_BINOMIAL_N, binomial_p=p,
                                theoretical_mean=_BINOMIAL_N*p,
                                theoretical_variance=_BINOMIAL_N*p*(1-p))
    comparison = {}
    for label, p, factor in [('middle_minus_coarse', probability, 2),
                              ('fine_minus_coarse', probability, 4),
                              ('fine_minus_middle', probability/2, 2)]:
        comparison[label] = dict(_moments(differences[label], -_BINOMIAL_N),
            unequal_count_pairs=int(differences[label].sum()-differences[label][_BINOMIAL_N]),
            theoretical_mean=0., theoretical_coupled_variance=_BINOMIAL_N*p*p*(1-1/factor),
            theoretical_independent_input_variance=_BINOMIAL_N*p*(2-p-p/factor))
    report = dict(status='COMPLETE_COUPLED_CLOCK_INPUT_DIAGNOSTIC', settings=settings,
        source_config_sha256=sha256_file(source_path), settings_sha256=sha256_file(config_path),
        implementation_sha256=sha256_file(Path(__file__)), source_drive=drive.metadata(),
        source_core_units=100000, coarse_ticks=coarse.ticks, external_input_duration_ms=config['duration_ms'],
        generated_counts=sum(v['count'] for v in marginals.values()), marginals=marginals,
        coupled_comparisons=comparison, elapsed_seconds=elapsed,
        python=platform.python_version(), numpy=np.__version__,
        initial_voltage_units='mV', initial_voltage_range=[float(initial.min()),float(initial.max())],
        source_simulations=0, production_episodes=0, confirmation_execution_released=False,
        confirmation_sampling_design_changed=False, numerical_margins_changed=False,
        scientific_fidelity_qualified=False, neural_clock_error_reduction_established=False,
        interpretation='Input-count diagnostic only. The within-input independent trials are not independent neural episodes. '
                       'Coarse source draw order is unchanged; fine clock seed trajectories differ from native same-seed runs. '
                       'The marginal law proof uses ideal independent auxiliary randomness; finite PRNG execution is numerically tested. '
                       'Input coupling does not remove integration, event-scheduling or nonlinear neural-response differences.',
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in (counts_path,rng_path)})
    freeze_json(output/'report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.config,args.project_root,args.output)
    print(json.dumps({key:report[key] for key in ('status','generated_counts','source_simulations')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

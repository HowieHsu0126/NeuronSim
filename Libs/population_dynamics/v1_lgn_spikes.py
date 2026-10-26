"""Explicit legacy LGN sampler dispatch on the supplied rate clock.

The source NumPy seed chain is retained inside one isolated RNG scope. Per-unit
seeds may repeat; the source Poisson branch changes subsequent seed draws.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable

import h5py
import numpy as np
import yaml

from .provenance import canonical_json, freeze_json, sha256_file, versions
from .v1_configured_inputs import (compare_population, delivered_file, read_h5_events,
                                   validate_plan)
from .v1_lgn_runtime import load_prepared_runtime
from .v1_nest_input_replay import (NEST_VERSION, bound_json, compare_events,
                                  execute_input_plan, load_pointnet_setter, prepare_inputs)
from .v1_nest_time_semantics import V1_DURATION_MS, V1_RESOLUTION_MS, encode_precise_times


def generate_source_spikes(time_s: np.ndarray, rates_hz: np.ndarray, source_ids: np.ndarray,
                           trials: int, starting_noise_seed: int,
                           primary_sampler: Callable, poisson_sampler: Callable
                           ) -> tuple[list[list[np.ndarray]], list[dict]]:
    """Return trials in source order and the exact seed/branch record.

    The actual supplied clock is used; the historical wrapper's reconstructed
    linspace is not used. Only the declared rate-domain condition selects the
    alternate sampler. Unexpected sampler errors propagate. The caller's global
    RNG states are restored after the complete invocation, including failures.
    """
    clock, rates, ids = map(np.asarray, (time_s, rates_hz, source_ids))
    if (clock.ndim != 1 or len(clock) < 2 or clock.dtype.kind != 'f'
            or not np.isfinite(clock).all() or clock[0] < 0 or not (np.diff(clock) > 0).all()
            or ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or (ids < 0).any() or len(np.unique(ids)) != len(ids)
            or rates.dtype.kind != 'f' or rates.shape != (len(ids), len(clock))
            or not np.isfinite(rates).all() or (rates < 0).any()):
        raise ValueError('SOURCE_SPIKE_CLOCK_RATES_OR_IDENTITIES_INVALID')
    if type(trials) is not int or trials < 1 or type(starting_noise_seed) is not int:
        raise ValueError('SOURCE_SPIKE_TRIAL_COUNT_AND_INTEGER_NOISE_SEED_REQUIRED')
    time_ms = clock * 1000.0
    start_ms, end_ms = min(clock)*1000.0, max(clock)*1000.0
    max_dt_ms = np.max(np.diff(time_ms))
    numpy_before, python_before = np.random.get_state(), random.getstate()
    outputs, seeds = [], []
    try:
        np.random.seed(starting_noise_seed)
        for trial in range(trials):
            trains = []
            for position, source_id in enumerate(ids):
                rate = rates[position]
                primary_seed = np.random.randint(10000)
                poisson_seed = None
                # The original primary seed is consumed even when its guard rejects.
                # Its guard includes the unused terminal rate sample and uses the
                # actual largest floating-point interval, not a nominal 1 ms.
                if np.max(rate) * max_dt_ms / 1000.0 > 0.1:
                    poisson_seed = np.random.randint(10000)
                    spikes = 1000.0 * np.asarray(poisson_sampler(clock, rate, seed=poisson_seed))
                    sampler = 'poisson'
                else:
                    spikes = np.asarray(primary_sampler(time_ms, rate, primary_seed,
                                                        start_ms, end_ms, 0.1))
                    sampler = 'bernoulli'
                if (spikes.ndim != 1 or spikes.dtype.kind != 'f'
                        or not np.isfinite(spikes).all() or (spikes < start_ms).any()
                        or (spikes > end_ms).any() or (np.diff(spikes) < 0).any()):
                    raise ValueError('SOURCE_SAMPLER_RETURNED_INVALID_EVENTS:' + str(source_id))
                trains.append(spikes)
                seeds.append(dict(trial=trial, source_id=int(source_id), sampler=sampler,
                                  primary_seed=int(primary_seed),
                                  poisson_seed=None if poisson_seed is None else int(poisson_seed)))
            outputs.append(trains)
    finally:
        np.random.set_state(numpy_before)
        random.setstate(python_before)
    return outputs, seeds


def input_plan_identity(plan):
    """Identify simulator input content independently of archive timestamps."""
    # NEST joins generators by source identity and sorts their event times.
    # Hash that delivered content, while leaving the original source order intact.
    order = np.lexsort((plan['source_id'], plan['population']))
    canonical_unit = np.empty(len(order), dtype=np.int64)
    canonical_unit[order] = np.arange(len(order))
    event_units = canonical_unit[plan['source_event_unit']]
    event_order = np.lexsort((plan['source_event_time_ms'], event_units))
    canonical = dict(population=plan['population'][order], source_id=plan['source_id'][order],
                     source_counts=plan['source_counts'][order],
                     source_event_unit=event_units[event_order],
                     source_event_time_ms=plan['source_event_time_ms'][event_order])
    digest = hashlib.sha256()
    for key, values in canonical.items():
        digest.update(canonical_json([key, values.dtype.str, list(values.shape)]).encode('utf-8'))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def write_spike_trial(source_ids, trains, seeds, duration_ms, stem, *, population_name='lgn'):
    """Persist source events and the complete inventory, then verify consumers.

    The event file contains active units only. The companion plan retains all
    source units and their seed/branch records, including silent units. A failed
    consumer check leaves the sampled events available for diagnosis.
    """
    if population_name not in ('lgn', 'bkg'):
        raise ValueError('SOURCE_SPIKE_POPULATION_NAME_INVALID')
    counts = np.asarray([len(train) for train in trains], dtype=np.int64)
    plan = dict(population=np.zeros(len(source_ids), dtype=np.int64),
                source_id=np.asarray(source_ids), source_counts=counts,
                source_event_unit=np.repeat(np.arange(len(source_ids)), counts),
                source_event_time_ms=np.concatenate(trains))
    if [row['source_id'] for row in seeds] != source_ids.tolist():
        raise ValueError('SOURCE_SPIKE_SEED_INVENTORY_DIFFERS')
    plan.update(source_primary_seed=np.asarray([r['primary_seed'] for r in seeds], dtype=np.int64),
                source_poisson_seed=np.asarray([-1 if r['poisson_seed'] is None else r['poisson_seed']
                                                for r in seeds], dtype=np.int64),
                source_sampler_code=np.asarray([int(r['sampler'] == 'poisson') for r in seeds], dtype=np.int8))
    with stem.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **plan)
    event_ids = plan['source_id'][plan['source_event_unit']]
    with h5py.File(stem.with_suffix('.h5'), 'x') as dest:
        group = dest.create_group('spikes/' + population_name)
        group.attrs['sorting'] = 'none'
        group.create_dataset('node_ids', data=event_ids)
        group.create_dataset('timestamps', data=plan['source_event_time_ms']).attrs['units'] = 'ms'
    validate_plan(plan)
    ids, times, metadata = read_h5_events(stem.with_suffix('.h5'), duration_ms)
    comparison = compare_population(plan, 0, ids, times)
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
        inventory_exact = set(saved.files) == set(plan) and all(
            saved[k].dtype == v.dtype and np.array_equal(saved[k], v) for k, v in plan.items())
    # The event comparator is multiset based. These checks also preserve the
    # source iteration order and the byte-level timestamp values of the sampler.
    order_exact = np.array_equal(ids, event_ids) and np.array_equal(times, plan['source_event_time_ms'])
    if (not inventory_exact or not order_exact or not comparison['all_event_pairs_exact']
            or metadata['source_population_labels'] != [population_name]):
        raise ValueError('SOURCE_SPIKE_SERIALIZATION_DIFFERS')
    return dict(input_identity_sha256=input_plan_identity(plan), plan=stem.with_suffix('.npz').name, sonata=stem.with_suffix('.h5').name,
                source_units=len(source_ids), silent_source_units=int((counts == 0).sum()),
                events=len(times), bernoulli_units=int((plan['source_sampler_code'] == 0).sum()),
                poisson_units=int((plan['source_sampler_code'] == 1).sum()),
                inventory_roundtrip_exact=inventory_exact, event_order_exact=bool(order_exact),
                event_comparison=comparison, sonata_metadata=metadata)


def export_source_spikes(rate_archive, config_path, output):
    """Run source samplers once on verified saved rates; do not regenerate rates."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='FAILED_SOURCE_SPIKE_EXPORT', runtime_environment=versions(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_rate_archive=str(rate_archive), source_fidelity_qualified=False,
                  original_visual_recipe_qualified=False, visual_action_mapping_qualified=False,
                  per_unit_noise_independence_claimed=False, production_episodes=0,
                  source_clock_policy='Unmodified saved rate clock, converted to ms for primary sampler.',
                  rng_policy='One starting NumPy seed per invocation; source seed chain in unit/trial order.',
                  sampler_codes={'0': 'bernoulli', '1': 'poisson'}, unused_poisson_seed=-1)
    try:
        reference, _ = bound_json(rate_archive, 'probe/probe.json')
        config = yaml.safe_load(config_path.read_text())
        if (reference['status'] != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
                or config['scope'] != 'development_source_runtime_qualification'
                or config['stimulus'] != reference['config']['stimulus']
                or config['unit_coverage'] != reference['config']['unit_coverage']
                or reference['time_units'] != 's' or reference['firing_rate_units'] != 'Hz'):
            raise ValueError('SOURCE_SPIKE_RATE_CASE_OR_UNITS_DIFFER')
        if 'input_construction_binding' in reference:
            report['input_construction_binding'] = reference['input_construction_binding']
        source_ids = np.asarray(reference['source_unit_ids'], dtype=np.int64)
        rate_path = delivered_file(rate_archive, 'probe/firing_rates_f_tot.h5')
        with h5py.File(rate_path, 'r') as source:
            clock, rates = source['time'][:], source['firing_rates_Hz'][:]
        if (list(rates.shape) != reference['rate_shape'] or len(source_ids) != reference['source_units']
                or clock[0] != 0.0 or clock[-1] != config['stimulus']['duration_s']):
            raise ValueError('SOURCE_SPIKE_RATE_INVENTORY_OR_CLOCK_DIFFERS')
        settings = config['spike_generation']
        report.update(config=config, config_sha256=sha256_file(config_path),
                      rate_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
                      rate_file_sha256=sha256_file(rate_path),
                      prepared_runtime_sha256=sha256_file(rate_archive/'prepared/prepared.json'),
                      source_units=len(source_ids), rate_shape=list(rates.shape),
                      starting_noise_seed=settings['starting_noise_seed'], trials=[])
        functions, poisson = load_prepared_runtime(rate_archive/'prepared')
        trains, seeds = generate_source_spikes(clock, rates, source_ids,
            settings['trials'], settings['starting_noise_seed'],
            functions.f_rate_to_spike_train, poisson.generate_inhomogenous_poisson)
        report['sampling_seconds'] = time.perf_counter() - started
        for trial, rows in enumerate(trains):
            selected_seeds = seeds[trial*len(source_ids):(trial+1)*len(source_ids)]
            result = write_spike_trial(source_ids, rows, selected_seeds,
                                      float(clock[-1]*1000.0), output/f'trial_{trial:04d}')
            report['trials'].append(dict(trial=trial, **result))
        report['status'] = 'SOURCE_SPIKES_EXPORTED_EXACT'
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifacts={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
        freeze_json(output/'report.json', report)
    return report


def load_spike_trial(archive, trial=0):
    """Read an exact exported trial with its full inventory and archive binding."""
    report, _ = bound_json(archive, 'spikes/report.json')
    rows = [row for row in report['trials'] if row['trial'] == trial]
    if report['status'] != 'SOURCE_SPIKES_EXPORTED_EXACT' or len(rows) != 1:
        raise ValueError('SOURCE_EXPORTED_SPIKE_TRIAL_REQUIRED')
    row = rows[0]
    path = delivered_file(archive, 'spikes/'+row['plan'])
    if sha256_file(path) != report['artifacts'][row['plan']]:
        raise ValueError('SOURCE_EXPORTED_SPIKE_PLAN_CHANGED')
    with np.load(path, allow_pickle=False) as source:
        plan = {key: source[key] for key in source.files}
    validate_plan(plan)
    identity = input_plan_identity(plan)
    if row.get('input_identity_sha256', identity) != identity:
        raise ValueError('SOURCE_EXPORTED_SPIKE_CONTENT_IDENTITY_DIFFERS')
    row['input_identity_sha256'] = identity
    if (set(np.unique(plan['population']).tolist()) != {0}
            or len(plan['source_id']) != row['source_units']
            or len(plan['source_event_time_ms']) != row['events']):
        raise ValueError('SOURCE_EXPORTED_SPIKE_INVENTORY_DIFFERS')
    # Native setters consume trains in inventory order, including silent units.
    units = plan['source_event_unit']
    expected_order = np.repeat(np.arange(len(plan['source_id'])), plan['source_counts'])
    if not np.array_equal(units, expected_order):
        raise ValueError('SOURCE_EXPORTED_SPIKE_UNIT_ORDER_DIFFERS')
    boundaries = np.cumsum(plan['source_counts'])[:-1]
    plan['trains'] = np.split(plan['source_event_time_ms'], boundaries)
    return plan, report


def replace_lgn_inputs(raw_inputs, generated, original_plan, origin):
    """Replace only the source-configured LGN trains in a fresh input object.

    Exact comparison to the previously observed raw plan binds population roles
    to the current source pickle. Edges and all other populations are reused.
    """
    baseline = prepare_inputs(raw_inputs, V1_DURATION_MS)
    for key in ('population', 'source_id', 'source_counts', 'source_event_unit', 'source_event_time_ms'):
        if not np.array_equal(baseline[key], original_plan[key]):
            raise ValueError('SOURCE_LGN_ORIGINAL_INPUT_BINDING_DIFFERS:'+key)
    if (origin['status'] != 'SOURCE_INPUT_CANDIDATES_EXACT'
            or origin['candidate_event_and_inventory_identity_exact'] is not True
            or origin['source_clock'] != dict(dt=V1_RESOLUTION_MS, duration=V1_DURATION_MS)):
        raise ValueError('SOURCE_LGN_POPULATION_ROLE_UNQUALIFIED')
    candidates = [row for row in origin['cases'] if row['target_node_set'] == 'lgn'
                  and row['all_event_pairs_exact'] and row['input_generator_inventory_exact']]
    if len(candidates) != 1:
        raise ValueError('SOURCE_LGN_POPULATION_ROLE_AMBIGUOUS')
    population = candidates[0]['candidate_archive_population']
    if type(population) is not int or not 0 <= population < len(raw_inputs):
        raise ValueError('SOURCE_LGN_POPULATION_INDEX_INVALID')
    nodes, edges = raw_inputs[population]
    validate_plan(generated)
    if (set(np.unique(generated['population']).tolist()) != {0}
            or not np.array_equal(np.sort(generated['source_id']), np.sort(nodes['ids']))):
        raise ValueError('SOURCE_LGN_COMPLETE_GENERATOR_INVENTORY_REQUIRED')
    by_id = {int(source_id): values for source_id, values in zip(generated['source_id'], generated['trains'])}
    replacement = dict(nodes, spikes=[by_id[int(source_id)] for source_id in nodes['ids']])
    result = list(raw_inputs)
    result[population] = (replacement, edges)
    merged = prepare_inputs(result, V1_DURATION_MS)
    for index in range(len(raw_inputs)):
        if index == population:
            continue
        # Unchanged objects retain background events and all synapse metadata.
        if result[index] is not raw_inputs[index]:
            raise ValueError('SOURCE_LGN_REPLACEMENT_CHANGED_ANOTHER_POPULATION')
    return result, dict(replaced_population=population, target_node_set='lgn',
        generated_units=len(generated['source_id']), generated_events=int(generated['source_counts'].sum()),
        retained_populations=[i for i in range(len(raw_inputs)) if i != population],
        original_events=int(baseline['source_counts'].sum()), merged_events=int(merged['source_counts'].sum()),
        background_events_unchanged=True, input_edge_objects_unchanged=True)


def replay_source_spikes(spike_archive, pointnet_source_archive, output):
    """Replay a generated trial using the existing NEST setter and recorder path."""
    import nest  # Optional native backend; export does not require NEST.

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='FAILED_GENERATED_INPUT_REPLAY', runtime_environment=versions(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  spike_archive=str(spike_archive), nest_version=nest.__version__,
                  source_fidelity_qualified=False, visual_action_mapping_qualified=False,
                  full_cortical_network_executed=False, production_episodes=0)
    try:
        if nest.__version__ != NEST_VERSION:
            raise ValueError('SOURCE_SPIKE_REPLAY_NEST_VERSION_DIFFERS')
        plan, source = load_spike_trial(spike_archive)
        if source['config']['stimulus']['duration_s']*1000.0 != V1_DURATION_MS:
            raise ValueError('SOURCE_SPIKE_REPLAY_DURATION_DIFFERS')
        report.update(source_spike_report_sha256=sha256_file(spike_archive/'spikes/report.json'),
                      source_units=len(plan['source_id']), source_events=int(plan['source_counts'].sum()),
                      source_config_clock=dict(dt=V1_RESOLUTION_MS, duration=V1_DURATION_MS),
                      observation_scope='Input generators connected only to the existing diagnostic recorder.')
        setter = load_pointnet_setter(pointnet_source_archive, nest)
        execute_input_plan(nest, plan, setter, output, report, V1_RESOLUTION_MS, V1_DURATION_MS, started)
        encoded = encode_precise_times(plan['source_event_time_ms'], float(nest.GetKernelStatus('tics_per_ms')))
        encoded_plan = dict(plan, source_event_time_ms=encoded['decoded_times_ms'])
        with np.load(output/'source_inputs.npz', allow_pickle=False) as saved:
            native_ids = saved['native_id']
        comparisons = {}
        for name in ('at_source_stop', 'after_recorder_flush'):
            with np.load(output/('events_'+name+'.npz'), allow_pickle=False) as saved:
                events = {key: saved[key] for key in saved.files}
            comparisons[name] = compare_events(encoded_plan, native_ids, events)
        report['source_encoded_event_comparisons'] = comparisons
        report['status'] = ('GENERATED_INPUT_SOURCE_ENCODING_EXACT'
                            if comparisons['after_recorder_flush']['all_event_pairs_exact']
                            else 'GENERATED_INPUT_DELIVERY_DIFFERS')
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifacts={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
        freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--rate-archive', type=Path)
    inputs.add_argument('--spike-archive', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--pointnet-source-archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rate_archive is not None:
        if args.config is None:
            parser.error('--config is required for rate export')
        report = export_source_spikes(args.rate_archive, args.config, args.output)
    else:
        if args.pointnet_source_archive is None:
            parser.error('--pointnet-source-archive is required for native replay')
        report = replay_source_spikes(args.spike_archive, args.pointnet_source_archive, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] in ('SOURCE_SPIKES_EXPORTED_EXACT', 'GENERATED_INPUT_SOURCE_ENCODING_EXACT') else 2


if __name__ == '__main__':
    raise SystemExit(main())

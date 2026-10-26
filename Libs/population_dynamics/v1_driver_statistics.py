"""Describe source V1 driver moments and seed reuse on verified saved rates.

Noise blocks are simulation repeats. This diagnostic does not establish biological
independence, production source fidelity, or equivalence to a continuous Poisson
LGN process. The primary source sampler is Bernoulli on its actual rate clock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.stats import t as student_t
import yaml

from .benchmark_tasks import condition_for_episode, v1_visual_config
from .circuit_campaign import campaign_config, episode_identities
from .provenance import freeze_json, sha256_file
from .v1_execution_recipe import require_same_recipe
from .resource_probe import probe
from .v1_circuit_stimulus import background_rates
from .v1_lgn_runtime import load_prepared_runtime
from .v1_lgn_spikes import generate_source_spikes
from .v1_nest_input_replay import bound_json


def source_count_moments(clock_s: np.ndarray, rates_hz: np.ndarray) -> dict:
    """Ideal branch-law moments; the finite-seed distribution is not enumerated."""
    clock, rates = map(np.asarray, (clock_s, rates_hz))
    if (clock.ndim != 1 or len(clock) < 2 or clock.dtype.kind != 'f'
            or rates.ndim != 2 or rates.shape[1] != len(clock)
            or rates.dtype.kind != 'f' or not np.isfinite(clock).all()
            or not np.isfinite(rates).all() or (rates < 0).any()
            or clock[0] != 0 or not (np.diff(clock) > 0).all()):
        raise ValueError('V1_DRIVER_FINITE_RATE_CLOCK_REQUIRED')
    intervals = np.diff(clock * 1000.0) / 1000.0
    probability = rates[:, :-1] * intervals
    poisson = rates.max(axis=1) * intervals.max() > 0.1
    if np.any(probability[~poisson] > 0.1):
        raise ValueError('V1_DRIVER_PRIMARY_BRANCH_DOMAIN_DIFFERS')
    mean = probability.sum(axis=1)
    variance = np.where(poisson, mean, (probability * (1.0 - probability)).sum(axis=1))
    return dict(mean=mean, variance=variance, poisson=poisson,
                terminal_rate_sample_in_count_expectation=False,
                terminal_rate_sample_in_branch_guard=True)


def seed_collision_summary(primary_seeds: np.ndarray) -> dict:
    """Count actual seed reuse. Repeated seeds do not prove identical events."""
    seeds = np.asarray(primary_seeds)
    if (seeds.ndim != 1 or seeds.dtype.kind not in 'iu'
            or (seeds < 0).any() or (seeds >= 10000).any()):
        raise ValueError('V1_DRIVER_SOURCE_PRIMARY_SEED_DOMAIN_REQUIRED')
    _, multiplicity = np.unique(seeds, return_counts=True)
    return dict(draws=len(seeds), unique_seeds=len(multiplicity),
                repeated_draws=len(seeds)-len(multiplicity),
                colliding_pairs=int(np.sum(multiplicity*(multiplicity-1)//2)),
                largest_multiplicity=int(multiplicity.max()) if len(multiplicity) else 0)



def block_seed_reuse(primary_seeds: np.ndarray, counts: np.ndarray,
                     source_ids: np.ndarray, noise_blocks: np.ndarray) -> dict:
    """Separate realized seed collisions within units, blocks and populations.

    Same-unit repeated seeds must give the same count for a fixed rate recipe.
    Equal counts alone do not prove equal event trains. Finite independent seed
    draws can collide, so these counts are not a test of independence.
    """
    seeds, values, ids, blocks = map(np.asarray,
                                    (primary_seeds, counts, source_ids, noise_blocks))
    if (seeds.ndim != 2 or not all(seeds.shape) or values.shape != seeds.shape
            or values.dtype.kind not in 'iu' or (values < 0).any()
            or ids.ndim != 1 or ids.shape != (seeds.shape[1],)
            or blocks.ndim != 1 or blocks.shape != (seeds.shape[0],)
            or ids.dtype.kind not in 'iu' or blocks.dtype.kind not in 'iu'
            or (ids < 0).any() or (blocks < 0).any()
            or len(np.unique(ids)) != len(ids)
            or len(np.unique(blocks)) != len(blocks)):
        raise ValueError('V1_DRIVER_BOUND_SEED_COUNT_MATRIX_REQUIRED')
    total = seed_collision_summary(seeds.ravel())
    same_block = sum(seed_collision_summary(row)['colliding_pairs'] for row in seeds)
    same_unit = 0
    units = []
    for unit in range(seeds.shape[1]):
        column = seeds[:, unit]
        reuse = seed_collision_summary(column)
        same_unit += reuse['colliding_pairs']
        repeated = []
        for seed in np.unique(column):
            positions = np.flatnonzero(column == seed)
            if len(positions) < 2:
                continue
            observed = values[positions, unit]
            if not np.all(observed == observed[0]):
                raise ValueError('V1_DRIVER_SAME_UNIT_SEED_COUNT_REPLAY_DIFFERS')
            repeated.append(dict(primary_seed=int(seed),
                noise_block_indices=blocks[positions].astype(int).tolist(),
                invocation_count=len(positions), count=int(observed[0])))
        units.append(dict(source_id=int(ids[unit]), **reuse, repeated_seed_groups=repeated))
    return dict(**total, same_unit_cross_block_seed_pairs=same_unit,
        cross_unit_seed_pairs=total['colliding_pairs']-same_unit,
        same_block_cross_unit_seed_pairs=same_block,
        cross_block_seed_pairs=total['colliding_pairs']-same_block,
        same_unit_repeated_seed_counts_exact=True, per_source_unit=units,
        event_train_equality_tested=False, distributional_independence_tested=False,
        interpretation='Realized finite-seed reuse only; no covariance or effective-sample-size estimate.')


def analyze_saved_noise(diagnostic: Path, output: Path) -> dict:
    """Read authenticated count arrays without repeating source invocations."""
    if not os.environ.get('SLURM_JOB_ID') or output.exists():
        raise ValueError('V1_DRIVER_FRESH_SLURM_EXECUTION_REQUIRED')
    report, delivery = bound_json(diagnostic, 'analysis/report.json')
    if delivery['exit_code'] != 0 or report['status'] != 'COMPLETE_SOURCE_DRIVER_DIAGNOSTIC_NOT_QUALIFIED':
        raise ValueError('V1_DRIVER_COMPLETE_SAVED_DIAGNOSTIC_REQUIRED')
    cases = []
    for case in report['cases']:
        path = 'analysis/'+case['name']+'_counts.npz'
        row = next(entry for entry in delivery['files'] if entry['path'] == path)
        array_path = diagnostic/path
        if array_path.stat().st_size != row['bytes'] or sha256_file(array_path) != row['sha256']:
            raise ValueError('V1_DRIVER_SAVED_COUNT_ARRAY_CHANGED')
        with np.load(array_path, allow_pickle=False) as arrays:
            summary = block_seed_reuse(arrays['primary_seeds'], arrays['counts'],
                                      arrays['source_ids'], arrays['noise_blocks'])
        cases.append(dict(name=case['name'], array_sha256=row['sha256'], seed_reuse=summary))
    result = dict(status='COMPLETE_SAVED_DRIVER_SEED_REUSE_NOT_QUALIFIED',
        source_diagnostic=str(diagnostic), source_report_sha256=sha256_file(diagnostic/'analysis/report.json'),
        moment_reference='ideal Bernoulli/Poisson branch law, not enumerated finite-seed distribution',
        cases=cases, source_invocations=0, cortical_simulations=0, production_episodes=0,
        scientific_qualification=False, distributional_independence_tested=False,
        recorded_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output, result)
    return result




SOURCE_PRIMARY_SEED_COUNT = 10000


def finite_seed_count_summary(class_counts: np.ndarray) -> dict:
    """Enumerated moments under uniform selection of the legal primary seed."""
    values=np.asarray(class_counts)
    if (values.ndim!=2 or values.shape[1]!=SOURCE_PRIMARY_SEED_COUNT
            or not values.shape[0] or values.dtype.kind not in 'iu' or (values<0).any()):
        raise ValueError('V1_DRIVER_COMPLETE_FINITE_SEED_COUNT_TABLE_REQUIRED')
    return dict(primary_seed_domain=[0,SOURCE_PRIMARY_SEED_COUNT-1],
        primary_seeds_per_class=SOURCE_PRIMARY_SEED_COUNT,
        mean_count=values.mean(axis=1).tolist(),count_variance=values.var(axis=1,ddof=0).tolist(),
        reference_law='uniform legal primary seed with fixed rate recipe',
        all_seeds_enumerated=True,joint_noise_law_qualified=False)


def finite_seed_count_lookup(class_counts: np.ndarray, rate_class: np.ndarray,
                             primary_seeds: np.ndarray) -> np.ndarray:
    """Reuse an enumerated count only for its exact rate class and primary seed."""
    table,classes,seeds=map(np.asarray,(class_counts,rate_class,primary_seeds))
    finite_seed_count_summary(table)
    if (classes.ndim!=1 or classes.dtype.kind not in 'iu' or (classes<0).any()
            or (classes>=len(table)).any() or seeds.ndim!=2
            or seeds.shape[1]!=len(classes) or seeds.dtype.kind not in 'iu'
            or (seeds<0).any() or (seeds>=SOURCE_PRIMARY_SEED_COUNT).any()):
        raise ValueError('V1_DRIVER_FINITE_SEED_LOOKUP_IDENTITIES_REQUIRED')
    return table[classes[None,:],seeds]


def finite_gray_joint_count_summary(class_counts: np.ndarray, rate_class: np.ndarray,
                                    counts: np.ndarray) -> dict:
    """Compare saved population counts with explicit finite-seed references.

    Independent uniform primary seeds give zero covariance between distinct
    units. Conditioning two seeds to be equal changes that reference. The equal
    and distinct seed covariances cancel under independent seed selection.
    Neither a realized collision nor empirical covariance is a test here.
    """
    table, classes, values = map(np.asarray, (class_counts, rate_class, counts))
    law = finite_seed_count_summary(table)
    if (classes.ndim != 1 or classes.dtype.kind not in 'iu' or not len(classes)
            or (classes < 0).any() or (classes >= len(table)).any()
            or set(classes.tolist()) != set(range(len(table)))
            or values.ndim != 2 or values.shape[1] != len(classes)
            or values.shape[0] < 2 or values.dtype.kind not in 'iu'
            or (values < 0).any()):
        raise ValueError('V1_DRIVER_FULL_JOINT_COUNT_IDENTITIES_REQUIRED')
    sizes = np.bincount(classes, minlength=len(table))
    totals = np.column_stack([values[:, classes == group].sum(axis=1)
                              for group in range(len(table))])
    centered = table.astype(np.float64) - np.asarray(law['mean_count'])[:, None]
    equal_seed_covariance = centered @ centered.T / SOURCE_PRIMARY_SEED_COUNT
    distinct_seed_covariance = -equal_seed_covariance / (SOURCE_PRIMARY_SEED_COUNT - 1)
    independent_covariance = np.diag(sizes * np.asarray(law['count_variance']))
    observed_covariance = np.atleast_2d(np.cov(totals, rowvar=False, ddof=1))
    individual_variance = values.var(axis=0, ddof=1)
    distinct_unit_sum = observed_covariance.copy()
    distinct_unit_sum[np.diag_indices(len(table))] -= np.bincount(
        classes, weights=individual_variance, minlength=len(table))
    pairs = sizes[:, None] * sizes[None, :]
    pairs[np.diag_indices(len(table))] -= sizes
    average_covariance = [[float(distinct_unit_sum[g, h] / pairs[g, h])
                           if pairs[g, h] else None for h in range(len(table))]
                          for g in range(len(table))]
    expected_variance = float(independent_covariance.sum())
    observed_variance = float(values.sum(axis=1).var(ddof=1))
    return dict(full_lgn_units=len(classes), noise_blocks=len(values),
        rate_class_population_sizes=sizes.tolist(),
        independent_primary_seed_reference=dict(
            law='Independent uniform primary seeds in [0,9999], fixed exact rate classes',
            source_model_total_mean=(sizes * np.asarray(law['mean_count'])).tolist(),
            source_model_total_covariance=independent_covariance.tolist(),
            total_count_variance=expected_variance),
        conditional_pair_references=dict(
            equal_primary_seed_covariance=equal_seed_covariance.tolist(),
            distinct_primary_seed_covariance=distinct_seed_covariance.tolist(),
            equal_primary_seed_probability=1.0 / SOURCE_PRIMARY_SEED_COUNT,
            interpretation='Conditional pair references; their independent-selection mixture has zero distinct-unit covariance.'),
        observed_source_model_total_covariance=observed_covariance.tolist(),
        observed_total_count_mean=float(values.sum(axis=1).mean()),
        observed_total_count_variance=observed_variance,
        observed_to_independent_total_variance_ratio=(observed_variance / expected_variance
                                                     if expected_variance else None),
        descriptive_mean_distinct_unit_count_covariance=average_covariance,
        distinct_unit_ordered_pairs=pairs.tolist(),
        scientific_equivalence='NOT_EVALUATED_NO_FROZEN_TOLERANCE',
        distributional_independence_tested=False, joint_noise_law_qualified=False,
        interpretation='Descriptive count covariance across saved starting-seed invocations. No significance, effective sample size, temporal covariance or cortical-impact claim.')


def analyze_saved_finite_gray(diagnostic: Path, output: Path) -> dict:
    """Analyze authenticated full-population counts without sampler execution."""
    if not os.environ.get('SLURM_JOB_ID') or output.exists():
        raise ValueError('V1_DRIVER_FRESH_SLURM_EXECUTION_REQUIRED')
    report, delivery = bound_json(diagnostic, 'analysis/report.json')
    if (delivery['exit_code'] != 0 or report['status'] !=
            'COMPLETE_FULL_GRAY_FINITE_PRIMARY_SEED_COUNT_LAW_NOT_QUALIFIED'
            or report['full_lgn_units'] != 17400 or report['full_count_realizations'] != 100):
        raise ValueError('V1_DRIVER_COMPLETE_SAVED_FINITE_GRAY_REQUIRED')
    row = next(entry for entry in delivery['files']
               if entry['path'] == 'analysis/finite_seed_counts.npz')
    path = diagnostic / row['path']
    if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
        raise ValueError('V1_DRIVER_SAVED_FINITE_COUNT_ARRAY_CHANGED')
    with np.load(path, allow_pickle=False) as arrays:
        table, classes, seeds, counts, ids, starts, totals = (
            arrays[key] for key in ('class_counts', 'rate_class', 'primary_seeds',
                                   'counts', 'source_ids', 'starting_noise_seeds',
                                   'source_model_total_counts'))
        if (ids.shape != (17400,) or ids.dtype.kind not in 'iu' or (ids < 0).any()
                or len(np.unique(ids)) != len(ids) or classes.shape != ids.shape
                or seeds.shape != (100, 17400) or counts.shape != seeds.shape
                or starts.shape != (100,) or starts.dtype.kind not in 'iu'
                or (starts < 0).any() or (starts > np.iinfo(np.uint32).max).any()
                or len(np.unique(starts)) != len(starts)
                or starts.tolist() != delivery['origins']['starting_noise_seeds']):
            raise ValueError('V1_DRIVER_SAVED_FULL_GRAY_IDENTITIES_DIFFER')
        if not np.array_equal(counts, finite_seed_count_lookup(table, classes, seeds)):
            raise ValueError('V1_DRIVER_SAVED_FULL_GRAY_COUNT_LOOKUP_DIFFERS')
        expected_seeds = np.asarray([np.random.RandomState(int(start)).randint(
            SOURCE_PRIMARY_SEED_COUNT, size=len(ids)) for start in starts])
        if not np.array_equal(seeds, expected_seeds):
            raise ValueError('V1_DRIVER_SAVED_FULL_GRAY_PRIMARY_SEEDS_DIFFER')
        summary = finite_gray_joint_count_summary(table, classes, counts)
        actual_totals = np.column_stack([counts[:, classes == g].sum(axis=1)
                                        for g in range(len(table))])
        if not np.array_equal(totals, actual_totals):
            raise ValueError('V1_DRIVER_SAVED_FULL_GRAY_POPULATION_TOTALS_DIFFER')
    result = dict(status='COMPLETE_SAVED_FULL_GRAY_JOINT_COUNT_REFERENCE_NOT_QUALIFIED',
        source_diagnostic=str(diagnostic), source_report_sha256=sha256_file(
            diagnostic / 'analysis/report.json'), source_array_sha256=row['sha256'],
        summary=summary, source_invocations=0, cortical_simulations=0,
        production_episodes=0, scientific_qualification=False,
        recorded_utc=datetime.now(timezone.utc).isoformat(), exit_code=0)
    freeze_json(output, result)
    return result


def run_finite_gray(config_path: Path, output: Path) -> dict:
    """Enumerate the unchanged full-gray primary sampler; produce counts only."""
    if not os.environ.get('SLURM_JOB_ID') or output.exists():
        raise ValueError('V1_DRIVER_FRESH_SLURM_EXECUTION_REQUIRED')
    config=yaml.safe_load(config_path.read_text())
    if (config['scope']!='full_gray_finite_seed_diagnostic' or config['training_seeds_used']!=[]
            or config['starting_noise_seeds']!=list(range(130000000,130000100))):
        raise ValueError('V1_DRIVER_FINITE_GRAY_DIAGNOSTIC_REQUIRED')
    rate_root=Path(config['rate_archive']);rate,delivery=bound_json(rate_root,'probe/probe.json')
    row=next(x for x in delivery['files'] if x['path']=='probe/firing_rates_f_tot.h5')
    if (delivery['exit_code']!=0 or rate['rate_shape']!=[17400,3001]
            or rate['config']['stimulus']['contrast_percent']!=0
            or sha256_file(rate_root/row['path'])!=row['sha256']):
        raise ValueError('V1_DRIVER_VERIFIED_FULL_GRAY_RATE_REQUIRED')
    with h5py.File(rate_root/row['path'],'r') as f:
        clock,rates=f['time'][:],f['firing_rates_Hz'][:]
    class_root=Path(config['rate_class_archive']);classes=json.loads((class_root/'report.json').read_text())
    if (classes['status']!='VERIFIED_FULL_GRAY_EXACT_RATE_REUSE_CLASSES'
            or classes['exit_code']!=0 or classes['rate_array_sha256']!=row['sha256']
            or classes['source_units']!=17400 or classes['exact_rate_classes']!=14):
        raise ValueError('V1_DRIVER_VERIFIED_EXACT_RATE_CLASSES_REQUIRED')
    class_row=next(x for x in classes['files'] if x['path']=='rate_classes.npz')
    if sha256_file(class_root/'rate_classes.npz')!=class_row['sha256']:
        raise ValueError('V1_DRIVER_RATE_CLASS_ARRAY_CHANGED')
    with np.load(class_root/'rate_classes.npz',allow_pickle=False) as f:
        ids=f['source_ids'];groups=f['rate_class_index']
    if ids.tolist()!=rate['source_unit_ids'] or set(groups.tolist())!=set(range(14)):
        raise ValueError('V1_DRIVER_FULL_GRAY_CLASS_IDENTITIES_DIFFER')
    representatives=[]
    for group in range(14):
        positions=np.flatnonzero(groups==group);first=int(positions[0])
        if not all(np.array_equal(rates[first],rates[x]) for x in positions):
            raise ValueError('V1_DRIVER_EXACT_RATE_CLASS_DIFFERS')
        representatives.append(first)
    moments=source_count_moments(clock,rates[representatives])
    if moments['poisson'].any():raise ValueError('V1_DRIVER_FINITE_PRIMARY_BRANCH_REQUIRED')
    spike_root=Path(config['full_gray_spike_archive'])
    spike,spike_delivery=bound_json(spike_root,'visual_inputs/report.json')
    lgn,_=bound_json(spike_root,'visual_inputs/lgn/report.json')
    if (spike_delivery['exit_code']!=0
            or spike['input_construction_binding']!=rate['input_construction_binding']
            or lgn['rate_report_sha256']!=sha256_file(rate_root/'probe/probe.json')):
        raise ValueError('V1_DRIVER_FINITE_GRAY_SOURCE_BINDING_DIFFERS')
    output.mkdir(parents=True);report=dict(status='FAILED_FINITE_GRAY_DIAGNOSTIC',
        complete_mechanistic_episodes=0,production_episodes=0,cortical_simulations=0,
        full_event_trains_saved=False,joint_noise_law_qualified=False,scientific_qualification=False,
        sampler_calls=0)
    numpy_state,python_state=np.random.get_state(),random.getstate()
    try:
        if probe(output/'resources.json',required_memory_gib=4)['status']!='PASS':
            raise RuntimeError('V1_DRIVER_FINITE_GRAY_RESOURCE_BLOCKED')
        functions,poisson=load_prepared_runtime(Path(config['prepared_archive'])/'prepared')
        class_counts=np.empty((14,SOURCE_PRIMARY_SEED_COUNT),dtype=np.int64)
        t=clock*1000.
        for group,representative in enumerate(representatives):
            for seed in range(SOURCE_PRIMARY_SEED_COUNT):
                trains=functions.f_rate_to_spike_train(t,rates[representative],seed,t[0],t[-1],.1)
                class_counts[group,seed]=len(trains);report['sampler_calls']+=1
            print(json.dumps(dict(rate_class_complete=group)),flush=True)
        seeds=np.asarray([np.random.RandomState(seed).randint(SOURCE_PRIMARY_SEED_COUNT,size=17400)
                          for seed in config['starting_noise_seeds']],dtype=np.int64)
        counts=finite_seed_count_lookup(class_counts,groups,seeds)
        plan_row=next(x for x in spike_delivery['files'] if x['path']=='visual_inputs/lgn/trial_0000.npz')
        if sha256_file(spike_root/plan_row['path'])!=plan_row['sha256']:
            raise ValueError('V1_DRIVER_FULL_GRAY_PLAN_CHANGED')
        with np.load(spike_root/plan_row['path'],allow_pickle=False) as p:
            if p['source_id'].tolist()!=ids.tolist() or p['source_sampler_code'].any():
                raise ValueError('V1_DRIVER_FULL_GRAY_PLAN_IDENTITIES_DIFFER')
            historical_counts=finite_seed_count_lookup(class_counts,groups,p['source_primary_seed'][None,:])[0]
            if not np.array_equal(historical_counts,p['source_counts']):
                raise ValueError('V1_DRIVER_FINITE_COUNTS_DIFFER_FROM_ALL_REAL_SOURCE_COUNTS')
            offsets=np.r_[0,np.cumsum(p['source_counts'])];checked=0
            for index in representatives:
                actual=np.asarray(functions.f_rate_to_spike_train(t,rates[index],int(p['source_primary_seed'][index]),t[0],t[-1],.1))
                expected=p['source_event_time_ms'][offsets[index]:offsets[index+1]]
                if not np.array_equal(actual,expected):raise ValueError('V1_DRIVER_REPRESENTATIVE_SOURCE_EVENTS_DIFFER')
                checked+=1
        law=finite_seed_count_summary(class_counts)
        population_counts=np.column_stack([counts[:,groups==g].sum(axis=1) for g in range(14)])
        np.savez_compressed(output/'finite_seed_counts.npz',class_counts=class_counts,
            source_ids=ids,rate_class=groups,starting_noise_seeds=np.asarray(config['starting_noise_seeds']),
            primary_seeds=seeds,counts=counts,source_model_total_counts=population_counts)
        report.update(status='COMPLETE_FULL_GRAY_FINITE_PRIMARY_SEED_COUNT_LAW_NOT_QUALIFIED',
            full_lgn_units=17400,full_count_realizations=100,full_count_realization_method='lookup of unchanged sampler counts for exact rate class and source-generated primary seed',
            actual_historical_source_counts_verified=17400,representative_event_trains_verified=checked,
            rate_array_sha256=row['sha256'],finite_seed_law=law,
            ideal_branch_mean= moments['mean'].tolist(),ideal_branch_variance=moments['variance'].tolist(),
            enumerated_to_ideal_mean_error=(np.asarray(law['mean_count'])-moments['mean']).tolist(),
            enumerated_to_ideal_variance_ratio=(np.asarray(law['count_variance'])/moments['variance']).tolist(),
            rate_class_population_sizes=np.bincount(groups).tolist(),
            descriptive_source_model_count_covariance=np.cov(population_counts,rowvar=False).tolist(),
            descriptive_total_count_mean=float(counts.sum(axis=1).mean()),
            descriptive_total_count_variance=float(counts.sum(axis=1).var(ddof=1)),
            noise_starting_seed_scope='declared diagnostic seed starts on historical source circuit; not production circuit identity',
            interpretation='Exact marginal count moments for uniform finite primary seed. Starting-seed PRNG joint law, cortical dynamics and scientific tolerances remain unqualified.',exit_code=0)
    except BaseException as e:
        report.update(error=str(e),exit_code=1);raise
    finally:
        np.random.set_state(numpy_state);random.setstate(python_state)
        freeze_json(output/'report.json',dict(report,recorded_utc=datetime.now(timezone.utc).isoformat()))
    return report


def envelope_count_summary(counts: np.ndarray, expected: np.ndarray) -> dict:
    """Describe the full condition vector over blocks without inferential claims."""
    counts, expected = map(np.asarray, (counts, expected))
    if (counts.ndim != 3 or counts.shape[0] < 2 or counts.shape[1] != 10
            or counts.shape[2] < 1 or counts.dtype.kind not in 'iu'
            or (counts < 0).any() or expected.shape != counts.shape
            or expected.dtype.kind != 'f' or not np.isfinite(expected).all()
            or (expected < 0).any()):
        raise ValueError('V1_DRIVER_TEN_CONDITION_COUNT_ENVELOPE_REQUIRED')
    totals = counts.sum(axis=2, dtype=np.int64).astype(np.float64)
    reference = expected.sum(axis=2)
    residuals = totals-reference
    result = dict(noise_blocks=counts.shape[0], conditions=10, source_units=counts.shape[2],
        total_count_mean_by_condition=totals.mean(axis=0).tolist(),
        expected_total_count_mean_by_condition=reference.mean(axis=0).tolist(),
        total_count_error_mean_by_condition=residuals.mean(axis=0).tolist(),
        descriptive_total_count_covariance=np.cov(totals,rowvar=False).tolist(),
        descriptive_count_error_covariance=np.cov(residuals,rowvar=False).tolist(),
        joint_distribution_qualified=False, independent_units_established=False,
        scientific_equivalence='NOT_EVALUATED_NO_FROZEN_TOLERANCE',
        interpretation='Empirical across-block covariance only; low-contrast direction varies by block; no significance or effective sample size.')
    return result


def run_condition_envelope(config_path: Path, output: Path) -> dict:
    """Execute missing representative drivers for the canonical rotated block."""
    if not os.environ.get('SLURM_JOB_ID') or output.exists():
        raise ValueError('V1_DRIVER_FRESH_SLURM_EXECUTION_REQUIRED')
    config=yaml.safe_load(config_path.read_text())
    campaign=campaign_config(Path(config['campaign_config']))
    if (config['scope']!='source_driver_condition_envelope' or config['circuit_index']!=1
            or config['noise_block_indices']!=list(range(100)) or config['training_seeds_used']!=[]):
        raise ValueError('V1_DRIVER_CANONICAL_ENVELOPE_REQUIRED')
    identities=[x for x in episode_identities(campaign,'v1_visual') if x['circuit_index']==1]
    if len(identities)!=1000:
        raise ValueError('V1_DRIVER_ONE_COMPLETE_CIRCUIT_ENVELOPE_REQUIRED')
    expected_conditions=[(condition_for_episode(campaign['scenarios']['v1_visual']['task_protocol'],
                                              'v1_visual',x),x) for x in identities]
    specs={}
    loaded=[]
    for case in config['rates']:
        root=Path(case['rate_archive']);report,delivery=bound_json(root,'probe/probe.json')
        row=next(x for x in delivery['files'] if x['path']=='probe/firing_rates_f_tot.h5')
        if (delivery['exit_code']!=0 or report['rate_shape']!=[14,3001]
                or report['config']['unit_coverage']!='model_representatives'
                or sha256_file(root/row['path'])!=row['sha256']):
            raise ValueError('V1_DRIVER_VERIFIED_ENVELOPE_RATE_REQUIRED')
        with h5py.File(root/row['path'],'r') as f:clock,rates=f['time'][:],f['firing_rates_Hz'][:]
        if loaded and (report['input_construction_binding']!=loaded[0][1]['input_construction_binding']
                or report['source_unit_ids']!=loaded[0][1]['source_unit_ids'] or not np.array_equal(clock,loaded[0][2])):
            raise ValueError('V1_DRIVER_ENVELOPE_SOURCE_IDENTITIES_DIFFER')
        key=(case['condition_id'],case.get('direction_deg',0))
        if key in specs:raise ValueError('V1_DRIVER_DUPLICATE_ENVELOPE_RATE')
        value=(root,report,clock,rates,source_count_moments(clock,rates))
        specs[key]=value;loaded.append(value)
    base_config=loaded[0][1]['config']
    for condition,identity in expected_conditions:
        key=(condition['condition_id'],condition.get('direction_deg',0))
        if key not in specs:
            raise ValueError('V1_DRIVER_COMPLETE_ROTATED_RATE_SET_REQUIRED')
        expected_recipe=v1_visual_config(base_config,
            campaign['scenarios']['v1_visual']['task_protocol'],identity)
        require_same_recipe(specs[key][1]['config'],expected_recipe,family='visual')
    expected_keys={(c['condition_id'],c.get('direction_deg',0)) for c,x in expected_conditions}
    if set(specs)!=expected_keys:raise ValueError('V1_DRIVER_COMPLETE_ROTATED_RATE_SET_REQUIRED')
    prior=Path(config['reused_diagnostic']);old,old_delivery=bound_json(prior,'analysis/report.json')
    config_row=next(x for x in old_delivery['files'] if x['path']=='original_config.yaml')
    if sha256_file(prior/'original_config.yaml')!=config_row['sha256']:
        raise ValueError('V1_DRIVER_REUSED_NOISE_CONFIG_CHANGED')
    old_config=yaml.safe_load((prior/'original_config.yaml').read_text())
    if (old_delivery['exit_code']!=0 or old_config['lgn_starting_seed']!=130001000
            or old_config['background_starting_seed']!=120001000):
        raise ValueError('V1_DRIVER_BOUND_REUSED_NOISE_REQUIRED')
    reuse={}
    for case in old['cases']:
        if case['input_construction_binding']!=loaded[0][1]['input_construction_binding']:
            raise ValueError('V1_DRIVER_REUSED_ACTUAL_CIRCUIT_DIFFERS')
        stimulus=case['stimulus'];cid={'direction_000':'direction_000','gray':'gray',
                                   'low_contrast_045':'low_contrast'}[case['name']]
        key=(cid,stimulus['direction_deg']);current=specs[key]
        if case['rate_array_sha256']!=sha256_file(current[0]/'probe/firing_rates_f_tot.h5'):
            raise ValueError('V1_DRIVER_REUSED_RATE_ARRAY_DIFFERS')
        path='analysis/'+case['name']+'_counts.npz';row=next(x for x in old_delivery['files'] if x['path']==path)
        if sha256_file(prior/path)!=row['sha256']:raise ValueError('V1_DRIVER_REUSED_COUNTS_CHANGED')
        with np.load(prior/path,allow_pickle=False) as z:
            if z['noise_blocks'].tolist()!=list(range(100)) or z['source_ids'].tolist()!=current[1]['source_unit_ids']:
                raise ValueError('V1_DRIVER_REUSED_BLOCK_IDS_DIFFER')
            reuse[key]=(z['counts'].copy(),z['primary_seeds'].copy())
    output.mkdir(parents=True)
    report=dict(status='FAILED_DRIVER_CONDITION_ENVELOPE',source_invocations=0,
        reused_lgn_invocations=0,cortical_simulations=0,production_episodes=0,scientific_qualification=False)
    try:
        if probe(output/'resources.json',required_memory_gib=2)['status']!='PASS':
            raise RuntimeError('V1_DRIVER_ENVELOPE_RESOURCE_BLOCKED')
        functions,poisson=load_prepared_runtime(loaded[0][0]/'prepared')
        counts=np.empty((100,10,14),dtype=np.int64);seeds=np.empty_like(counts)
        expected=np.empty(counts.shape,dtype=np.float64);variance=np.empty_like(expected)
        ids=np.asarray(loaded[0][1]['source_unit_ids']);conditions=[]
        for condition,identity in expected_conditions:
            b,j=identity['block_index'],identity['condition_index'];key=(condition['condition_id'],condition.get('direction_deg',0))
            root,rate,clock,values,moments=specs[key]
            stimulus=rate['config']['stimulus']
            if stimulus['direction_deg']!=condition.get('direction_deg',0) or stimulus['contrast_percent']!=condition['contrast_percent']:
                raise ValueError('V1_DRIVER_CANONICAL_CONDITION_DIFFERS')
            if key in reuse:
                counts[b,j]=reuse[key][0][b];seeds[b,j]=reuse[key][1][b];report['reused_lgn_invocations']+=1
            else:
                trains,record=generate_source_spikes(clock,values,ids,1,identity['stimulus_seed'],
                    functions.f_rate_to_spike_train,poisson.generate_inhomogenous_poisson)
                counts[b,j]=[len(x) for x in trains[0]];seeds[b,j]=[x['primary_seed'] for x in record]
                report['source_invocations']+=1
            expected[b,j]=moments['mean'];variance[b,j]=moments['variance']
            conditions.append(dict(block_index=b,condition_index=j,condition=condition,
                lgn_starting_seed=identity['stimulus_seed'],background_starting_seed=identity['background_seed']))
            if j==9 and b%10==0:print(json.dumps(dict(block_complete=b)),flush=True)
        bkg_path='analysis/background_counts.npz';row=next(x for x in old_delivery['files'] if x['path']==bkg_path)
        if sha256_file(prior/bkg_path)!=row['sha256']:raise ValueError('V1_DRIVER_REUSED_BACKGROUND_CHANGED')
        with np.load(prior/bkg_path,allow_pickle=False) as z:
            if z['noise_blocks'].tolist()!=list(range(100)):raise ValueError('V1_DRIVER_BACKGROUND_BLOCKS_DIFFER')
            background=z['counts'].copy();background_seeds=z['source_seeds'].copy()
        np.savez_compressed(output/'condition_counts.npz',counts=counts,primary_seeds=seeds,
            source_ids=ids,noise_blocks=np.arange(100),condition_indices=np.arange(10),
            expected_count=expected,expected_variance=variance,background_counts=background,background_seeds=background_seeds)
        diagnostics=[]
        for j in range(10):
            for direction in sorted({x['condition'].get('direction_deg',0) for x in conditions if x['condition_index']==j}):
                blocks=np.asarray([x['block_index'] for x in conditions if x['condition_index']==j and x['condition'].get('direction_deg',0)==direction])
                key=(next(x['condition']['condition_id'] for x in conditions if x['condition_index']==j),direction)
                diagnostics.append(dict(condition_index=j,direction_deg=direction,noise_block_indices=blocks.tolist(),
                    count_diagnostics=count_diagnostics(counts[blocks,j],specs[key][4])))
        report.update(status='COMPLETE_REPRESENTATIVE_DRIVER_CONDITION_ENVELOPE_NOT_QUALIFIED',
            canonical_condition_noise_pairs=1000,source_units=14,background_invocations_reused=100,
            low_contrast_direction_block_counts={str(d):sum(x['condition_index']==9 and x['condition']['direction_deg']==d for x in conditions) for d in range(0,360,45)},
            conditions=conditions,diagnostics=diagnostics,summary=envelope_count_summary(counts,expected),
            one_circuit_only=True,production_rotated_condition_envelope_tested_on_representatives=True,
            full_lgn_population_tested=False,joint_distribution_qualified=False,exit_code=0)
    except BaseException as e:
        report.update(error=str(e),exit_code=1);raise
    finally:
        freeze_json(output/'report.json',dict(report,recorded_utc=datetime.now(timezone.utc).isoformat()))
    return report


def count_diagnostics(counts: np.ndarray, moments: dict) -> list[dict]:
    """Whole-invocation mean intervals; time bins and units are not repeats."""
    values = np.asarray(counts)
    if (values.ndim != 2 or values.dtype.kind not in 'iu' or len(values) < 2
            or (values < 0).any() or values.shape[1] != len(moments['mean'])):
        raise ValueError('V1_DRIVER_COMPLETE_COUNT_REPEAT_MATRIX_REQUIRED')
    result = []
    for index in range(values.shape[1]):
        x = values[:, index].astype(np.float64)
        mean, variance = float(x.mean()), float(x.var(ddof=1))
        expected_mean, expected_variance = float(moments['mean'][index]), float(moments['variance'][index])
        radius = float(student_t.ppf(.975, len(x)-1) * np.sqrt(variance/len(x)))
        result.append(dict(unit_index=index, noise_blocks=len(x), mean_count=mean,
            mean_count_interval_95=[mean-radius, mean+radius],
            mean_interval_method='Student_t_whole_source_invocations_approximate_not_simultaneous',
            observed_count_variance=variance, source_expected_count=expected_mean,
            source_expected_count_variance=expected_variance,
            mean_count_error=mean-expected_mean,
            variance_ratio=None if expected_variance == 0 else variance/expected_variance,
            source_branch='poisson' if moments['poisson'][index] else 'bernoulli',
            scientific_equivalence='NOT_EVALUATED_NO_FROZEN_TOLERANCE'))
    return result


def run(config_path: Path, output: Path) -> dict:
    if (not os.environ.get('SLURM_JOB_ID') or output.exists()):
        raise ValueError('V1_DRIVER_FRESH_SLURM_EXECUTION_REQUIRED')
    config = yaml.safe_load(config_path.read_text())
    blocks = config['noise_block_indices']
    if (config['scope'] != 'source_driver_diagnostic' or blocks != list(range(100))
            or config['training_seeds_used'] != [] or config['circuit_index'] != 1):
        raise ValueError('V1_DRIVER_DECLARED_DIAGNOSTIC_SCOPE_REQUIRED')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status='FAILED_SOURCE_DRIVER_DIAGNOSTIC', complete_episodes=0,
        production_episodes=0, cortical_simulations=0, scientific_qualification=False,
        biological_independence_established=False, training_seeds_used=[], cases=[])
    try:
        if probe(output/'resources.json', required_memory_gib=2)['status'] != 'PASS':
            raise RuntimeError('V1_DRIVER_RESOURCE_BLOCKED')
        lgn_start = config['lgn_starting_seed']
        bkg_start = config['background_starting_seed']
        loaded = []
        for case in config['cases']:
            root = Path(case['rate_archive'])
            rate, delivery = bound_json(root, 'probe/probe.json')
            if (delivery['exit_code'] != 0 or rate['source_units'] != 14
                    or rate['rate_shape'] != [14,3001] or rate['config']['unit_coverage'] != 'model_representatives'):
                raise ValueError('V1_DRIVER_VERIFIED_REPRESENTATIVE_RATE_REQUIRED')
            row = next(x for x in delivery['files'] if x['path']=='probe/firing_rates_f_tot.h5')
            if sha256_file(root/row['path']) != row['sha256']:
                raise ValueError('V1_DRIVER_RATE_BYTES_CHANGED')
            with h5py.File(root/row['path'], 'r') as arrays:
                clock, rates = arrays['time'][:], arrays['firing_rates_Hz'][:]
            if loaded and rate['input_construction_binding'] != loaded[0][2]['input_construction_binding']:
                raise ValueError('V1_DRIVER_SAME_ACTUAL_CIRCUIT_REQUIRED')
            loaded.append((case, root, rate, clock, rates))
        functions, poisson = load_prepared_runtime(loaded[0][1]/'prepared')
        # Use source samplers unchanged; no vectorized replacement of random draws.
        for case, root, rate, clock, rates in loaded:
            counts, seeds = [], []
            for block in blocks:
                outputs, record = generate_source_spikes(clock, rates, np.asarray(rate['source_unit_ids']),
                    1, lgn_start+block, functions.f_rate_to_spike_train, poisson.generate_inhomogenous_poisson)
                counts.append([len(train) for train in outputs[0]])
                seeds.append([entry['primary_seed'] for entry in record])
            counts, seeds = np.asarray(counts,dtype=np.int64), np.asarray(seeds,dtype=np.int64)
            np.savez_compressed(output/(case['name']+'_counts.npz'),counts=counts,primary_seeds=seeds,
                source_ids=np.asarray(rate['source_unit_ids']),noise_blocks=np.asarray(blocks))
            report['cases'].append(dict(name=case['name'],rate_archive=str(root),
                rate_report_sha256=sha256_file(root/'probe/probe.json'),rate_array_sha256=sha256_file(root/'probe/firing_rates_f_tot.h5'),
                input_construction_binding=rate['input_construction_binding'],
                stimulus=rate['config']['stimulus'],source_units=14,
                fixed_condition_noise_invocations=100,low_contrast_rotation_reproduced=False,
                count_diagnostics=count_diagnostics(counts,source_count_moments(clock,rates)),
                across_block_primary_seed_reuse=seed_collision_summary(seeds.ravel())))
            print(json.dumps(dict(case=case['name'],noise_invocations=len(counts))),flush=True)
        settings=dict(loaded[0][2]['config']['background']);clock,rates=background_rates(settings,3000.)
        counts, seeds = [], []
        for block in blocks:
            outputs, record=generate_source_spikes(clock,rates,np.asarray([0]),1,bkg_start+block,
                functions.f_rate_to_spike_train,poisson.generate_inhomogenous_poisson)
            counts.append([len(outputs[0][0])]);seeds.append([record[0]['primary_seed'],record[0]['poisson_seed']])
        counts,seeds=np.asarray(counts,dtype=np.int64),np.asarray(seeds,dtype=np.int64)
        np.savez_compressed(output/'background_counts.npz',counts=counts,source_seeds=seeds,noise_blocks=np.asarray(blocks))
        report['background']=dict(count_diagnostics=count_diagnostics(counts,source_count_moments(clock,rates)),
            primary_seed_reuse=seed_collision_summary(seeds[:,0]),poisson_seed_reuse=seed_collision_summary(seeds[:,1]),
            shared_across_conditions=True)
        # Check collisions on the actual complete historical gray input, not an assumed seed law.
        root=Path(config['full_gray_spike_archive']);bundle,delivery=bound_json(root,'visual_inputs/report.json')
        path=root/'visual_inputs/lgn/trial_0000.npz';row=next(x for x in delivery['files'] if x['path']=='visual_inputs/lgn/trial_0000.npz')
        if sha256_file(path)!=row['sha256'] or not bundle['complete_lgn_population']:
            raise ValueError('V1_DRIVER_FULL_GRAY_SOURCE_BINDING_REQUIRED')
        with np.load(path,allow_pickle=False) as arrays:
            actual=arrays['source_primary_seed'];branch=arrays['source_sampler_code']
            if len(actual)!=17400 or np.any(branch):
                raise ValueError('V1_DRIVER_FULL_GRAY_EXPECTED_ALL_PRIMARY_BRANCH_REQUIRED')
            generator=np.random.RandomState(bundle['lgn_noise_seed']);expected=generator.randint(10000,size=17400)
            if not np.array_equal(actual,expected):
                raise ValueError('V1_DRIVER_FULL_GRAY_SOURCE_SEED_REPLAY_DIFFERS')
            report['full_gray_seed_reuse']=dict(**seed_collision_summary(actual),seed_chain_exact=True,
                source_units=17400,spike_archive=str(root),plan_sha256=row['sha256'],
                interpretation='Seed reuse exists in the source. Event correlation requires separate measurement; no independent-unit-noise claim.')
        report.update(status='COMPLETE_SOURCE_DRIVER_DIAGNOSTIC_NOT_QUALIFIED',exit_code=0,
            simulation_repeat_unit='whole_source_noise_invocation',
            mean_interval_assumption='approximate independent invocation counts; source finite seed reuse can create dependence; no significance claim',
            represented_scope='one canary circuit, three fixed conditions and one background process; full gray seed record is historical',
            production_rotated_condition_envelope_tested=False,source_seed_policy_changed=False)
    except BaseException as error:
        report.update(error=str(error),exit_code=1)
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',type=Path)
    source.add_argument('--saved-diagnostic',type=Path)
    source.add_argument('--condition-envelope-config',type=Path)
    source.add_argument('--finite-gray-config',type=Path)
    source.add_argument('--saved-finite-gray',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.saved_finite_gray:
        result=analyze_saved_finite_gray(args.saved_finite_gray,args.output)
    elif args.finite_gray_config:
        result=run_finite_gray(args.finite_gray_config,args.output)
    elif args.condition_envelope_config:
        result=run_condition_envelope(args.condition_envelope_config,args.output)
    elif args.saved_diagnostic:
        result=analyze_saved_noise(args.saved_diagnostic,args.output)
    else:
        result=run(args.config,args.output)
    print(json.dumps(result))


if __name__=='__main__':
    main()

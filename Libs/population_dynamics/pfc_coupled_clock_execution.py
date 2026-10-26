"""Complete PFC fine-clock development episodes using coupled source inputs.

The pinned numerical engine is unchanged. New fine stochastic realizations
have explicit provenance and never inherit native same-seed qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as executor
from . import pfc_native_drive as native_drive
from . import pfc_source_execution as native_execution
from . import pfc_clock_coupling, pfc_coupled_drive, pfc_clock_temporal_control
from .pfc_coupled_drive import create_coupled_drive
from .pfc_delay_analysis import CONDITIONS, write_rows
from .pfc_fluctuation_analysis import unit_mapping
from .pfc_scale_analysis import _source_bound_file, _source_manifest, _source_phase_rows
from .pfc_temporal_readouts import temporal_rows
from .provenance import freeze_json, sha256_file

GAIN_SCHEMA = 'pfc_coupled_clock_recurrent_gain_v1'
GAIN_BLOCK_SCHEMA = 'pfc_coupled_clock_recurrent_gain_block_v1'
GAIN_SCOPE = 'development_coupled_clock_recurrent_gain_pilot'
GAIN_KERNEL = dict(sigma_weight_profile=20.0, Jpos_excit2excit=2.3)
GAIN_COHORTS = {
    'reference_2560': (2048, 512, [93001, 93002, 93003], 98711),
    'expanded_100k': (80000, 20000, [95001, 95002, 95003], 95411),
}

# The original pilot registration remains fixed. This separate development bank
# fills only the uncomputed clocks for the existing coarse-replication blocks.
GAIN_REPLICATES_SCHEMA = 'pfc_coupled_clock_recurrent_gain_replicates_v1'
GAIN_REPLICATES_BLOCK_SCHEMA = 'pfc_coupled_clock_recurrent_gain_replicates_block_v1'
GAIN_REPLICATES_SCOPE = 'development_coupled_clock_recurrent_gain_replicates'
GAIN_REPLICATES_COHORTS = {
    'reference_2560': (2048, 512, list(range(93004, 93033)), 98717),
    'expanded_100k': (80000, 20000, list(range(95004, 95033)), 95417),
}
GAIN_REGISTRIES = {
    GAIN_SCHEMA: (GAIN_BLOCK_SCHEMA, GAIN_SCOPE, GAIN_COHORTS),
    GAIN_REPLICATES_SCHEMA: (GAIN_REPLICATES_BLOCK_SCHEMA, GAIN_REPLICATES_SCOPE,
                             GAIN_REPLICATES_COHORTS),
}
GAIN_BLOCK_REGISTRIES = {block_schema: (scope, cohorts)
                         for block_schema, scope, cohorts in GAIN_REGISTRIES.values()}


REFINEMENT_REPEAT_SCHEMA = 'pfc_coupled_refinement_repeat_v1'
REFINEMENT_REPEAT_BLOCK_SCHEMA = 'pfc_coupled_refinement_repeat_block_v1'
REFINEMENT_REPEAT_SCOPE = 'development_coupled_refinement_repeat'

TEMPORAL_CONTROL_SCHEMA = 'pfc_coupled_clock_temporal_control_v1'
TEMPORAL_CONTROL_SCOPE = 'development_within_parent_temporal_order_control'
TEMPORAL_CONTROL_TRANSFORM = 'within_coarse_tick_tree_temporal_reversal'
TEMPORAL_REFERENCE_FILES = ('report.json', 'initial_voltage.npy', 'final_rng.npz',
                            'applied_current_observation.npz', 'applied_current_metadata.json')



REFERENCE_FILES = ('report.json', 'initial_voltage.npy', 'final_rng.npz',
                   'applied_current_observation.npz', 'applied_current_metadata.json',
                   'units.npz', 'unit_metadata.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _gain_block(settings):
    scope, cohorts = GAIN_BLOCK_REGISTRIES[settings['schema']]
    require(settings.get('scope') == scope
            and settings.get('confirmation_execution_released') is False
            and settings.get('cohort') in cohorts,
            'REGISTERED_RECURRENT_GAIN_DEVELOPMENT_COHORT_REQUIRED')
    ne, ni, seeds, auxiliary_start = cohorts[settings['cohort']]
    seed = settings.get('source_rng_seed')
    require(type(seed) is int and seed in seeds
            and settings.get('excitatory_neurons') == ne
            and settings.get('inhibitory_neurons') == ni
            and settings.get('kernel_parameters') == GAIN_KERNEL,
            'REGISTERED_RECURRENT_GAIN_SIZE_SEED_AND_KERNEL_REQUIRED')
    first = auxiliary_start + 2 * seeds.index(seed)
    require(settings.get('refinement_seeds') == [first, first + 1],
            'REGISTERED_RECURRENT_GAIN_AUXILIARY_SEEDS_REQUIRED')
    qualifications = settings.get('fine_clock_qualifications', {})
    require(set(qualifications) == {'0.025', '0.0125'}
            and all(isinstance(row, dict) and isinstance(row.get('archive'), str)
                    and row['archive'] for row in qualifications.values()),
            'EXPLICIT_FINE_CLOCK_QUALIFICATION_ARCHIVES_REQUIRED')


def _refinement_repeat_block(settings):
    seed=settings.get('source_rng_seed');aux=settings.get('refinement_seeds', [])
    require(settings.get('scope') == REFINEMENT_REPEAT_SCOPE
            and type(seed) is int and 95001 <= seed <= 95032
            and settings.get('confirmation_execution_released') is False
            and settings.get('auxiliary_repeat_index') == 1
            and settings.get('new_original_independent_blocks') == 0,
            'REGISTERED_REFINEMENT_REPEAT_BLOCK_REQUIRED')
    original=[95411+2*(seed-95001),95412+2*(seed-95001)]
    require(settings.get('original_refinement_seeds') == original
            and len(aux) == 2 and len(set(aux)) == 2
            and all(type(x) is int and 0 <= x < 2**32 for x in aux)
            and not set(aux) & set(range(95411,95475))
            and not set(aux) & set(range(96001,96129)),
            'NEW_AUXILIARY_STREAMS_OUTSIDE_FROZEN_AND_CONFIRMATION_RANGES_REQUIRED')
    require(settings.get('conditions') == ['cue','omission','strong_cue']
            and settings.get('fine_dt_ms') == [.025,.0125] and settings.get('coarse_dt_ms') == .05
            and set(settings.get('coarse_references',{})) == set(settings['conditions']),
            'COMPLETE_REFINEMENT_REPEAT_CLOCK_GRID_REQUIRED')



def _temporal_control_block(settings):
    require(settings.get('scope') == TEMPORAL_CONTROL_SCOPE
            and settings.get('source_rng_seed') == 95001
            and settings.get('refinement_seeds') == [95411, 95412]
            and settings.get('temporal_transform') == TEMPORAL_CONTROL_TRANSFORM
            and settings.get('new_original_independent_blocks') == 0
            and settings.get('new_auxiliary_streams') == 0
            and settings.get('new_development_episodes') == 6
            and settings.get('maximum_reserved_confirmation_blocks') == 128
            and settings.get('confirmation_execution_released') is False
            and settings.get('production_episodes') == 0,
            'REGISTERED_SINGLE_BLOCK_TEMPORAL_CONTROL_REQUIRED')
    require(settings.get('conditions') == ['cue', 'omission', 'strong_cue']
            and settings.get('fine_dt_ms') == [.025, .0125]
            and settings.get('coarse_dt_ms') == .05
            and set(settings.get('coarse_references', {})) == set(settings['conditions']),
            'COMPLETE_TEMPORAL_CONTROL_CLOCK_GRID_REQUIRED')
    reference = settings.get('original_fine_reference', {})
    require(isinstance(reference.get('archive'), str) and bool(reference['archive'])
            and isinstance(reference.get('delivery_sha256'), str)
            and len(reference['delivery_sha256']) == 64,
            'BOUND_ORIGINAL_FINE_REALIZATION_REQUIRED')


def load_temporal_reference(settings, project_root, dt_ms, condition, config, case):
    """Bind original A before a temporal control; do not read its spike arrays."""
    reference = settings['original_fine_reference']
    root = project_root / reference['archive']
    require(sha256_file(root/'delivery.json') == reference['delivery_sha256'],
            'ORIGINAL_FINE_DELIVERY_CHANGED')
    delivery = json.loads((root/'delivery.json').read_text())
    require(delivery.get('verified') is True
            and delivery.get('status') == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT',
            'COMPLETE_ORIGINAL_FINE_PILOT_REQUIRED')
    manifest = _source_manifest(delivery)
    prefix = Path('source_cases') / case_key(dt_ms, condition, settings['source_rng_seed'])
    paths = {name: _source_bound_file(root, manifest, str(prefix/name))
             for name in TEMPORAL_REFERENCE_FILES}
    original = json.loads(paths['report.json'].read_text())
    require(original.get('status') == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE'
            and original.get('source_execution_complete') is True
            and original.get('source_rng_seed') == settings['source_rng_seed']
            and original.get('refinement_seeds') == settings['refinement_seeds']
            and original.get('dt_ms') == dt_ms and original.get('condition') == condition
            and original.get('resolved_config') == config and original.get('selected_case') == case
            and not original.get('drive', {}).get('coupling', {}).get('input_order_control'),
            'EXACT_UNTRANSFORMED_ORIGINAL_FINE_CASE_REQUIRED')
    # The consumer factory changes only to admit the declared row adapter.
    # Numerical engine, native driver and original splitting kernel stay bound.
    for module in (executor, native_drive, native_execution, pfc_clock_coupling):
        name = module.__name__.split('.')[-1]
        require(original['code'][name] == sha256_file(Path(module.__file__)),
                'ORIGINAL_FINE_NUMERICAL_SOURCE_CHANGED:'+name)
    for name in TEMPORAL_REFERENCE_FILES:
        if name != 'report.json':
            require(original['outputs'][name]['sha256'] == sha256_file(paths[name]),
                    'ORIGINAL_FINE_OUTPUT_BINDING_CHANGED:'+name)
    return original, paths


def temporal_outcome_checks(initial, states, current_arrays, drive, original, paths):
    with np.load(paths['final_rng.npz'], allow_pickle=False) as saved:
        rng_equal = set(states) == set(saved.files) and all(
            np.array_equal(states[name], saved[name]) for name in saved.files)
    with np.load(paths['applied_current_observation.npz'], allow_pickle=False) as saved:
        current_equal = set(current_arrays) == set(saved.files) and all(
            np.array_equal(current_arrays[name], saved[name]) for name in saved.files)
    return dict(
        initial_voltage_equal=np.array_equal(initial, np.load(paths['initial_voltage.npy'], allow_pickle=False)),
        all_coarse_and_auxiliary_final_rng_equal=bool(rng_equal),
        full_fine_clock_applied_current_equal=bool(current_equal),
        population_external_event_totals_equal=drive.event_totals == original['drive']['external_event_totals'],
    )


def case_grid(settings):
    if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA:
        _temporal_control_block(settings)
        return [(dt, condition) for dt in settings['fine_dt_ms'] for condition in settings['conditions']]
    if settings.get('schema') == REFINEMENT_REPEAT_BLOCK_SCHEMA:
        _refinement_repeat_block(settings)
        return [(dt, condition) for dt in settings['fine_dt_ms'] for condition in settings['conditions']]
    if settings.get('schema') in GAIN_BLOCK_REGISTRIES:
        _gain_block(settings)
        require(settings.get('conditions') == ['cue', 'omission', 'strong_cue']
                and settings.get('fine_dt_ms') == [0.025, 0.0125]
                and settings.get('coarse_dt_ms') == 0.05
                and set(settings.get('coarse_references', {})) == set(settings['conditions']),
                'COMPLETE_RECURRENT_GAIN_CLOCK_GRID_REQUIRED')
        return [(dt, condition) for dt in settings['fine_dt_ms'] for condition in settings['conditions']]
    pilot = (settings.get('schema') == 'pfc_coupled_clock_pilot_v1'
             and settings.get('scope') == 'development_paired_clock_pilot'
             and settings.get('source_rng_seed') == 95001)
    calibration = (settings.get('schema') == 'pfc_coupled_clock_calibration_block_v1'
                   and settings.get('scope') == 'development_paired_clock_calibration'
                   and type(settings.get('source_rng_seed')) is int
                   and 95002 <= settings['source_rng_seed'] <= 95032)
    require((pilot or calibration) and settings.get('confirmation_execution_released') is False,
            'DEVELOPMENT_COUPLED_CLOCK_PILOT_REQUIRED')
    require(settings['conditions'] == ['cue', 'omission', 'strong_cue']
            and settings['fine_dt_ms'] == [0.025, 0.0125]
            and settings['coarse_dt_ms'] == 0.05
            and set(settings['coarse_references']) == set(settings['conditions']),
            'COMPLETE_FIRST_CALIBRATION_BLOCK_CLOCK_GRID_REQUIRED')
    return [(dt, condition) for dt in settings['fine_dt_ms'] for condition in settings['conditions']]



def select_settings(settings, source_rng_seed=None, cohort=None):
    """Select a registered block without changing legacy pilot/calibration calls."""
    if settings.get('schema') == REFINEMENT_REPEAT_SCHEMA:
        seeds=list(range(95001,95033));aux=settings.get('auxiliary_rng_seeds', [])
        require(cohort is None and settings.get('scope') == REFINEMENT_REPEAT_SCOPE
                and settings.get('source_rng_seeds') == seeds
                and settings.get('original_independent_blocks') == 32
                and settings.get('new_original_independent_blocks') == 0
                and settings.get('new_development_episodes') == 192
                and settings.get('maximum_reserved_confirmation_blocks') == 128
                and settings.get('production_episodes') == 0,
                'COMPLETE_EXISTING_32_BLOCK_REFINEMENT_REPEAT_REQUIRED')
        blocks=settings.get('blocks', [])
        require([b.get('source_rng_seed') for b in blocks] == seeds
                and len(aux) == 64 and len(set(aux)) == 64,
                'COMPLETE_ORDERED_BLOCKS_AND_UNIQUE_AUXILIARY_STREAMS_REQUIRED')
        selected=None
        shared={k:settings[k] for k in ('conditions','coarse_dt_ms','fine_dt_ms',
            'confirmation_execution_released','auxiliary_repeat_index','new_original_independent_blocks')}
        for i,block in enumerate(blocks):
            require(block.get('refinement_seeds') == aux[2*i:2*i+2],
                    'FROZEN_REFINEMENT_REPEAT_STREAM_ASSIGNMENT_REQUIRED')
            candidate=dict(shared,**block,schema=REFINEMENT_REPEAT_BLOCK_SCHEMA,scope=REFINEMENT_REPEAT_SCOPE)
            case_grid(candidate)
            if block['source_rng_seed'] == source_rng_seed:selected=candidate
        require(type(source_rng_seed) is int and selected is not None,'REGISTERED_REFINEMENT_REPEAT_SELECTION_REQUIRED')
        return selected
    if settings.get('schema') in GAIN_REGISTRIES:
        block_schema, scope, cohorts = GAIN_REGISTRIES[settings['schema']]
        require(settings.get('scope') == scope
                and settings.get('confirmation_execution_released') is False
                and set(settings.get('cohorts', {})) == set(cohorts),
                'COMPLETE_RECURRENT_GAIN_COHORTS_REQUIRED')
        shared = {key: settings[key] for key in
                  ('conditions', 'coarse_dt_ms', 'fine_dt_ms', 'confirmation_execution_released',
                   'kernel_parameters', 'fine_clock_qualifications')}
        selected = None
        for name, (ne, ni, seeds, _) in cohorts.items():
            bank = settings['cohorts'][name]
            require(bank.get('source_rng_seeds') == seeds
                    and bank.get('excitatory_neurons') == ne
                    and bank.get('inhibitory_neurons') == ni
                    and [block.get('source_rng_seed') for block in bank.get('blocks', [])] == seeds,
                    'EXACT_THREE_RECURRENT_GAIN_BLOCKS_PER_SIZE_REQUIRED' if settings['schema'] == GAIN_SCHEMA
                    else 'EXACT_REMAINING_29_RECURRENT_GAIN_BLOCKS_PER_SIZE_REQUIRED')
            for block in bank['blocks']:
                candidate = dict(shared, **block)
                candidate.update(schema=block_schema, scope=scope, cohort=name,
                                 excitatory_neurons=ne, inhibitory_neurons=ni)
                case_grid(candidate)
                if name == cohort and block['source_rng_seed'] == source_rng_seed:
                    selected = candidate
        require(type(source_rng_seed) is int and selected is not None,
                'REGISTERED_RECURRENT_GAIN_COHORT_AND_BLOCK_SELECTION_REQUIRED')
        return selected
    require(cohort is None, 'LEGACY_COHORT_OVERRIDE_FORBIDDEN')
    if settings.get('schema') != 'pfc_coupled_clock_calibration_v1':
        require(source_rng_seed is None, 'PILOT_BLOCK_OVERRIDE_FORBIDDEN')
        case_grid(settings)
        return settings
    require(settings.get('scope') == 'development_paired_clock_calibration'
            and settings.get('confirmation_execution_released') is False
            and settings.get('source_rng_seeds') == list(range(95002, 95033))
            and settings.get('reused_source_rng_seed') == 95001,
            'FIXED_REMAINING_CALIBRATION_BLOCKS_REQUIRED')
    blocks = settings['blocks']
    require(len(blocks) == 31 and [b['source_rng_seed'] for b in blocks] == settings['source_rng_seeds'],
            'COMPLETE_UNIQUE_ORDERED_CALIBRATION_BLOCKS_REQUIRED')
    for block in blocks:
        seed = block['source_rng_seed']
        require(block['refinement_seeds'] == [95411+2*(seed-95001), 95412+2*(seed-95001)]
                and set(block['coarse_references']) == set(settings['conditions']),
                'REGISTERED_AUXILIARY_SEEDS_AND_ALL_CONDITIONS_REQUIRED')
    require(type(source_rng_seed) is int and source_rng_seed in settings['source_rng_seeds'],
            'REGISTERED_CALIBRATION_BLOCK_SELECTION_REQUIRED')
    block = blocks[source_rng_seed-95002]
    selected = {key: settings[key] for key in
                ('conditions', 'coarse_dt_ms', 'fine_dt_ms', 'confirmation_execution_released')}
    selected.update(block, schema='pfc_coupled_clock_calibration_block_v1',
                    scope=settings['scope'])
    case_grid(selected)
    return selected


def select_array_seed(settings, task_index, cohort=None):
    """Resolve an array task through the same complete registry gate as the CLI."""
    require(type(task_index) is int, 'INTEGER_ARRAY_TASK_REQUIRED')
    if settings.get('schema') == REFINEMENT_REPEAT_SCHEMA:
        require(cohort is None, 'REFINEMENT_REPEAT_COHORT_OVERRIDE_FORBIDDEN')
        seeds=settings.get('source_rng_seeds', [])
    else:
        require(settings.get('schema') == GAIN_REPLICATES_SCHEMA
                and cohort in settings.get('cohorts', {}), 'REGISTERED_ARRAY_SCHEMA_AND_COHORT_REQUIRED')
        seeds=settings['cohorts'][cohort]['source_rng_seeds']
    require(0 <= task_index < len(seeds), 'REGISTERED_ARRAY_BLOCK_REQUIRED')
    seed=seeds[task_index]
    select_settings(settings, seed, cohort)
    return seed


def case_key(dt_ms, condition, seed):
    clock = str(dt_ms).replace('.', 'p')
    return f'dt_{clock}/pfc_{seed}_{condition}'


def load_coarse_inputs(settings, project_root, condition):
    """Bind an existing coarse episode; this does not qualify a fine clock."""
    case_grid(settings)
    require(condition in settings['conditions'], 'REGISTERED_COARSE_CONDITION_REQUIRED')
    reference = settings['coarse_references'][condition]
    root = project_root / reference['archive']
    require(sha256_file(root/'delivery.json') == reference['delivery_sha256'], 'COARSE_SOURCE_DELIVERY_CHANGED')
    delivery = json.loads((root/'delivery.json').read_text())
    require(delivery['status'] == 'ARCHIVED_PFC_SOURCE_EXECUTION'
            and delivery['result_status'] == 'PFC_SOURCE_EXECUTION_COMPLETE'
            and delivery['source_execution_complete'] is True, 'COMPLETE_COARSE_SOURCE_REQUIRED')
    manifest = _source_manifest(delivery)
    paths = {name: _source_bound_file(root, manifest, 'source_run/'+name) for name in REFERENCE_FILES}
    require(sha256_file(paths['report.json']) == reference['report_sha256'], 'COARSE_SOURCE_REPORT_CHANGED')
    source = json.loads(paths['report.json'].read_text())
    config = dict(source['resolved_config'])
    case = dict(source['selected_case'])
    expected_case = f"pfc_{settings['source_rng_seed']}_{condition}"
    gain = settings.get('schema') in GAIN_BLOCK_REGISTRIES
    ne = settings['excitatory_neurons'] if gain else 80000
    ni = settings['inhibitory_neurons'] if gain else 20000
    require(source['status'] == 'PFC_SOURCE_EXECUTION_COMPLETE'
            and source['source_execution_complete'] is True
            and source['backend'] == 'numpy' and source['case_id'] == expected_case
            and case['case_id'] == expected_case
            and source['seeds'] == {'source_rng_seed': settings['source_rng_seed']}
            and config['excitatory_neurons'] == ne and config['inhibitory_neurons'] == ni
            and config['dt_ms'] == settings['coarse_dt_ms'] and config['duration_ms'] == 5000.
            and config['readout_bin_ms'] == 5. and config['excitatory_preference_bins'] == 16
            and case['stimulus_strength_nA'] == CONDITIONS[condition]
            and case['distractor_strength_nA'] == 0.
            and all(row['passed'] is True for row in source['validations'].values()),
            'FROZEN_COMPLETE_REGISTERED_COARSE_CASE_REQUIRED' if gain else 'FROZEN_COMPLETE_100K_COARSE_CASE_REQUIRED')
    modules = [('executor_sha256', executor), ('drive_sha256', native_drive),
               ('runner_sha256', native_execution)]
    if gain:
        require(source['driver']['numpy_version'] == np.__version__, 'COARSE_SOURCE_NUMPY_VERSION_CHANGED')
        modules.append(('kernel_parameterization_sha256', native_execution.kernel_parameters))
        registered_case, _, dimensions = native_execution._case_config(config, expected_case)
        require(registered_case == case and dimensions == source['dimensions']
                and native_execution.kernel_parameters.resolve_kernel_parameters(config) == settings['kernel_parameters'],
                'COARSE_RECURRENT_GAIN_SOURCE_REGISTRATION_CHANGED')
    for key, module in modules:
        require(source['code'][key] == sha256_file(Path(module.__file__)), 'PINNED_SOURCE_CODE_CHANGED:'+key)
    for name in REFERENCE_FILES:
        if name != 'report.json':
            require(source['outputs'][name]['sha256'] == sha256_file(paths[name]), 'COARSE_OUTPUT_BINDING_CHANGED:'+name)
    return config, case, source, paths


def qualify_fine_clock(settings, project_root, dt_ms, config):
    """Use the source runner's existing code, NumPy, clock and kernel gate."""
    require((dt_ms, settings['conditions'][0]) in case_grid(settings), 'REGISTERED_FINE_CLOCK_REQUIRED')
    reference = settings['fine_clock_qualifications'][str(dt_ms)]
    directory = project_root / reference['archive']
    require((directory/'delivery.json').is_file()
            and (directory/'comparison/qualification.json').is_file(),
            'SUCCESSFUL_FINE_CLOCK_QUALIFICATION_REQUIRED:'+str(dt_ms))
    return native_execution._qualification(directory, 'numpy', dt_ms, config)


def prepare_coupled_initial(config, case, settings, paths, dt_ms):
    config = dict(config, dt_ms=dt_ms)
    factory = (pfc_clock_temporal_control.create_temporal_reversal_drive
               if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA else create_coupled_drive)
    initial, drive = factory(config, case, settings['coarse_dt_ms'], settings['refinement_seeds'])
    require(np.array_equal(initial, np.load(paths['initial_voltage.npy'], allow_pickle=False)),
            'SOURCE_INITIAL_VOLTAGE_REUSE_FAILED')
    return config, initial, drive


def load_inputs(settings, project_root, dt_ms, condition):
    require((dt_ms, condition) in case_grid(settings), 'REGISTERED_COUPLED_CLOCK_CASE_REQUIRED')
    config, case, source, paths = load_coarse_inputs(settings, project_root, condition)
    if settings.get('schema') in GAIN_BLOCK_REGISTRIES:
        source['coupled_fine_clock_qualification'] = qualify_fine_clock(settings, project_root, dt_ms, config)
    config, initial, drive = prepare_coupled_initial(config, case, settings, paths, dt_ms)
    if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA:
        original, original_paths = load_temporal_reference(settings, project_root, dt_ms, condition, config, case)
        require(np.array_equal(initial, np.load(original_paths['initial_voltage.npy'], allow_pickle=False)),
                'ORIGINAL_FINE_INITIAL_VOLTAGE_CHANGED')
        source['temporal_original_report'] = original
        source['temporal_original_paths'] = original_paths
    return config, case, initial, drive, source, paths


def source_dimensions(config):
    ne, ni = config['excitatory_neurons'], config['inhibitory_neurons']
    return dict(n_e=ne, n_i=ni, coupled_core_units=ne + ni)


def write_source_readouts(output, raw, config, case_id, source_rng_seed, condition):
    dimensions = source_dimensions(config)
    rows, scalar_rows, vectors, scalar_checks = _source_phase_rows(
        raw, case_id, {case_id: (source_rng_seed, condition)}, config, dimensions)
    write_rows(output/'phase_responses.csv', rows)
    write_rows(output/'phase_scalar_checks.csv', scalar_rows)
    np.savez(output/'phase_unit_counts.npz', **vectors)
    temporal = temporal_rows(raw['spike_ticks'], raw['spike_unit_indices'],
                             n_e=dimensions['n_e'], n_i=dimensions['n_i'], dt_ms=config['dt_ms'],
                             bin_ms=config['readout_bin_ms'], duration_ms=config['duration_ms'])
    write_rows(output/'temporal_readouts.csv', temporal)
    np.savez(output/'field_observations.npz', **field_observations(raw, config))
    return dict(scalar_phase_checks=scalar_checks, temporal_bins=len(temporal))


def current_agreement(actual, reference, fine_dt_ms, coarse_dt_ms):
    require(set(actual) == set(reference), 'APPLIED_CURRENT_SCHEMA_CHANGED')
    checks = {}
    for key in actual:
        left, right = actual[key], reference[key]
        if key in ('segment_start_ticks', 'segment_stop_ticks'):
            left = left * fine_dt_ms
            right = right * coarse_dt_ms
        checks[key] = bool(np.array_equal(left, right))
    return checks


def field_observations(raw, config):
    ne, ni = config['excitatory_neurons'], config['inhibitory_neurons']
    groups, sizes = unit_mapping(ne, ni, config['excitatory_preference_bins'])
    bin_ms = config['readout_bin_ms']; bin_ticks = round(bin_ms/config['dt_ms'])
    bins = round(config['duration_ms']/bin_ms)
    require(bin_ticks > 0 and bin_ticks*config['dt_ms'] == bin_ms
            and bins*bin_ms == config['duration_ms'], 'COMPLETE_SOURCE_FIELD_BINS_REQUIRED')
    ids, ticks = raw['spike_unit_indices'], raw['spike_ticks']
    require(ids.shape == ticks.shape and ids.ndim == 1 and ids.dtype.kind in 'iu'
            and ticks.dtype.kind in 'iu' and np.all((ids >= 0) & (ids < ne+ni))
            and np.all((ticks >= 0) & (ticks < bins*bin_ticks)), 'VALID_SOURCE_FIELD_EVENTS_REQUIRED')
    flat = (ticks//bin_ticks)*len(sizes)+groups[ids]
    counts = np.bincount(flat, minlength=bins*len(sizes)).reshape(bins, len(sizes))
    require(int(counts.sum()) == len(ids), 'FIELD_COUNTS_MUST_PRESERVE_ALL_SPIKES')
    return dict(field_counts=counts, field_unit_counts=sizes,
                field_indices=np.arange(len(sizes), dtype=np.int64),
                time_start_ms=np.arange(bins)*bin_ms,
                rates_hz=counts/(sizes[None, :]*(bin_ms/1000)))


def retag_engine_metadata(result):
    used, cost = result['inputs_used'], result['cost']
    used['engine_input_mode_tag'] = used['input_mode']
    used['input_mode'] = 'coupled_binomial_clock_refinement_stream'
    cost['coupled_input_generation_seconds'] = cost.pop('native_input_generation_seconds')
    arithmetic = cost['arithmetic_path']
    arithmetic['engine_external_input_tag'] = arithmetic['external_inputs']
    arithmetic['external_inputs'] = 'unchanged native coarse stream with independently seeded binomial half-clock refinement'


def execute_case(settings, config_path, project_root, output, dt_ms, condition):
    config, case, initial, drive, source, paths = load_inputs(settings, project_root, dt_ms, condition)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING_COUPLED_CLOCK_DEVELOPMENT_CASE', scope=settings['scope'],
        case_id=case['case_id'], condition=condition, source_rng_seed=settings['source_rng_seed'],
        refinement_seeds=settings['refinement_seeds'], dt_ms=dt_ms, coarse_dt_ms=settings['coarse_dt_ms'],
        resolved_config=config, selected_case=case, coarse_reference=settings['coarse_references'][condition],
        settings_sha256=sha256_file(config_path), coupled_core_units=source_dimensions(config)['coupled_core_units'],
        duration_ms=config['duration_ms'],
        source_execution_complete=False, native_fine_seed_sequence_preserved=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        production_episodes=0, original_independent_blocks=1,
        code={module.__name__.split('.')[-1]:sha256_file(Path(module.__file__)) for module in
              (executor, native_drive, native_execution, pfc_clock_coupling, pfc_coupled_drive)},
        runner_sha256=sha256_file(Path(__file__)))
    if settings.get('schema') in GAIN_BLOCK_REGISTRIES:
        report.update(cohort=settings['cohort'], kernel_parameters=settings['kernel_parameters'],
            fine_clock_qualification=source['coupled_fine_clock_qualification'],
            numpy_version=np.__version__, refinement_seeds_independent_units=False,
            clock_sensitivity_scope='coupled source-input laws at native discrete clocks; not an identical continuous event list or pure integration error')
    if settings.get('schema') == REFINEMENT_REPEAT_BLOCK_SCHEMA:
        report.update(auxiliary_repeat_index=1, original_refinement_seeds=settings['original_refinement_seeds'],
            original_independent_blocks=0, reused_original_independent_blocks=1,
            new_original_independent_blocks=0, refinement_seeds_independent_units=False)
    if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA:
        report.update(status='RUNNING_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE',
            temporal_transform=settings['temporal_transform'],
            original_fine_reference=settings['original_fine_reference'],
            original_fine_report_sha256=sha256_file(source['temporal_original_paths']['report.json']),
            original_independent_blocks=0, reused_original_independent_blocks=1,
            new_original_independent_blocks=0, new_auxiliary_streams=0,
            rng_streams_replayed=True, independent_auxiliary_repeat=False,
            refinement_seeds_independent_units=False)
        report['code']['pfc_clock_temporal_control'] = sha256_file(Path(pfc_clock_temporal_control.__file__))
    print(json.dumps(dict(status=report['status'], case_id=case['case_id'], dt_ms=dt_ms)), flush=True)
    try:
        np.save(output/'initial_voltage.npy', initial, allow_pickle=False)
        (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        for name in ('units.npz', 'unit_metadata.json'):
            shutil.copy2(paths[name], output/name)
        result = executor.simulate(config, case, initial, backend='numpy', drive=drive,
                                   observe_applied_current=True)
        # Preserve all computed arrays even if the following integrity checks fail.
        np.savez(output/'raw.npz', **result['raw'])
        np.savez(output/'terminal_state.npz', **result['terminal_state'])
        observation = result['applied_current_observation']
        np.savez(output/'applied_current_observation.npz', **observation['arrays'])
        freeze_json(output/'applied_current_metadata.json', observation['metadata'])
        states = drive.final_state_arrays()
        np.savez(output/'final_rng.npz', **states)
        validation = native_execution.validate_result(config, result, drive)
        report['validations'] = validation
        require(all(row['passed'] is True for row in validation.values()), 'FULL_SOURCE_OUTPUT_VALIDATION_FAILED')
        with np.load(paths['final_rng.npz'], allow_pickle=False) as expected:
            coarse_keys = {key.removeprefix('coarse__') for key in states if key.startswith('coarse__')}
            coarse_equal = coarse_keys == set(expected.files) and all(
                np.array_equal(states['coarse__'+key], expected[key]) for key in expected.files)
        report['coarse_final_rng_matches_archived_episode'] = bool(coarse_equal)
        require(coarse_equal, 'COARSE_SOURCE_FINAL_RNG_REUSE_FAILED')
        with np.load(paths['applied_current_observation.npz'], allow_pickle=False) as stored:
            reference_current = {key:stored[key] for key in stored.files}
        current_checks = current_agreement(observation['arrays'], reference_current, dt_ms, settings['coarse_dt_ms'])
        report['coarse_current_agreement'] = current_checks
        require(all(current_checks.values()), 'PHYSICAL_CONTROL_DIFFERS_FROM_COARSE_REFERENCE')
        retag_engine_metadata(result)
        if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA:
            checks = temporal_outcome_checks(initial, states, observation['arrays'], drive,
                source['temporal_original_report'], source['temporal_original_paths'])
            report['original_fine_input_agreement'] = checks
            require(all(checks.values()), 'TEMPORAL_CONTROL_ORIGINAL_INPUT_INVARIANT_FAILED')
            result['inputs_used']['input_order_control'] = drive.metadata()['coupling']['input_order_control']
            result['cost']['arithmetic_path']['external_inputs'] = 'original coupled source draws; child time rows reversed within each coarse tick'
        report.update(inputs_used=result['inputs_used'], cost=result['cost'], drive=drive.metadata())
        readouts = write_source_readouts(output, result['raw'], config, case['case_id'],
                                         settings['source_rng_seed'], condition)
        report.update(status=('COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE'
                             if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA
                             else 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE'), source_execution_complete=True,
            completed_duration_ticks=drive.ticks, development_mechanistic_episodes=1,
            **readouts,
            observation='full ascending source unit IDs;16source-preference E fields plus I;5ms bins;counts and Hz',
            source_clock_sensitivity_qualified=False, confidence_intervals=False)
    except Exception as exc:
        report.update(status=('FAILED_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE'
                             if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA
                             else 'FAILED_COUPLED_CLOCK_DEVELOPMENT_CASE'),
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report['outputs'] = {path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path))
                             for path in output.iterdir() if path.is_file() and path.name != 'report.json'}
        report.update(wall_seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/'report.json', report)
    print(json.dumps({key:report[key] for key in ('status','case_id','dt_ms','completed_duration_ticks')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--source-rng-seed', type=int)
    parser.add_argument('--cohort', choices=tuple(GAIN_COHORTS))
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--validate-inputs-only', action='store_true')
    validation.add_argument('--validate-coarse-only', action='store_true')
    args = parser.parse_args()
    settings = select_settings(yaml.safe_load(args.config.read_text()), args.source_rng_seed, args.cohort)
    grid = case_grid(settings)
    if args.validate_coarse_only:
        for condition in settings['conditions']:
            config, case, _, paths = load_coarse_inputs(settings, args.project_root, condition)
            for dt in settings['fine_dt_ms']:
                prepare_coupled_initial(config, case, settings, paths, dt)
                if settings.get('schema') == TEMPORAL_CONTROL_SCHEMA:
                    load_temporal_reference(settings, args.project_root, dt, condition,
                                            dict(config, dt_ms=dt), case)
        print(json.dumps(dict(status='COUPLED_CLOCK_COARSE_ANCHORS_VALIDATED',
            coarse_episodes=len(settings['conditions']), source_simulations=0,
            fine_clock_qualification_verified=False)))
        return 0
    if args.validate_inputs_only:
        for dt, condition in grid:load_inputs(settings, args.project_root, dt, condition)
        print(json.dumps(dict(status='COUPLED_CLOCK_INPUTS_VALIDATED', cases=len(grid), source_simulations=0)))
        return 0
    index = args.case_index if args.case_index is not None else int(os.environ['SLURM_PROCID'])
    require(0 <= index < len(grid), 'REGISTERED_COUPLED_CASE_INDEX_REQUIRED')
    dt, condition = grid[index]
    execute_case(settings, args.config, args.project_root,
                 args.output_root/case_key(dt, condition, settings['source_rng_seed']), dt, condition)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

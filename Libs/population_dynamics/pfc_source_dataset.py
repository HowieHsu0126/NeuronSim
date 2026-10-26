"""Convert completed, observed PFC source archives to development CircuitDataset data.

No simulator is called. The two source current commands and their source-clock
activation flags remain the control representation. Per-neuron delivered
currents and microscopic states are verification data, excluded from the normal
causal window interface. This entry point never creates production data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import yaml

from .circuit_data import CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits, SCHEMA
from .pfc_scale_analysis import (_source_registration, _source_header, _source_values,
    _source_bound_file, observed_source_registration, SOURCE_REVISION, SOURCE_ENTRY_SHA256, READOUT_RTOL, READOUT_ATOL)
from .provenance import canonical_json, sha256_file
from .pfc_distractor_response import registration as distractor_registration
from .pfc_kernel_parameters import circuit_provenance


VERIFY_WINDOW_BINS = 100


def _bound(header, name):
    return _source_bound_file(header['archive'], header['manifest'], 'source_run/'+name,
                              header['report']['outputs'][name])


def _arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _units(config, header, dimensions):
    source = _arrays(_bound(header, 'units.npz'))
    n_e, n_i = dimensions['n_e'], dimensions['n_i']
    fields = config['excitatory_preference_bins']
    if type(fields) is not int or not 1 <= fields <= n_e:
        raise ValueError('PFC_DATASET_PREFERENCE_FIELD_COUNT_INVALID')
    provenance = circuit_provenance(config, SOURCE_ENTRY_SHA256)
    parameter = provenance['parameter_identity']
    groups = np.r_[source['source_unit_indices'][:n_e]*fields//n_e, np.full(n_i,fields)]
    return CircuitUnits('pfc_'+parameter[:16], source['unit_ids'],
        np.where(source['population_index']==0, 'E', 'I'), groups,
        tuple([f'E_preference_{i*360/fields:g}_{(i+1)*360/fields:g}_deg' for i in range(fields)]+['I_population']),
        'E angle=360*source_index/N_E degrees; I has no preference coordinate', SOURCE_REVISION,
        circuit_provenance=provenance)


def _controls(config, case, observed):
    cue, distractor = observed['cue_active'], observed['distractor_active']
    values = np.column_stack((cue*case['stimulus_strength_nA'], distractor*case['distractor_strength_nA'],
                              cue, distractor)).astype(np.float64)
    features = (
        dict(name='cue_current', unit='nA', population='E',
             center_degree=config['stimulus_center_deg'], width_degree=config['stimulus_width_deg'],
             role='source_assignment_amplitude'),
        dict(name='distractor_current', unit='nA', population='E',
             center_degree=config.get('distractor_center_deg',90.), width_degree=config.get('distractor_width_deg',40.),
             role='source_assignment_amplitude'),
        dict(name='cue_active', unit='dimensionless', role='derived_source_schedule_indicator'),
        dict(name='distractor_active', unit='dimensionless', role='derived_source_schedule_indicator'))
    # A zero-amplitude active cue differs from an inactive cue in the source's
    # current-reset operation; the activation flags must remain explicit.
    return observed['segment_start_ticks'], values, features


def _exact(stored, expected, label):
    if stored.shape != expected.shape or stored.dtype != expected.dtype:
        raise ValueError('PFC_DATASET_ROUNDTRIP_SCHEMA_MISMATCH:'+label)
    for first in range(0,len(expected),65536):
        if not np.array_equal(stored[first:first+65536], expected[first:first+65536]):
            raise ValueError('PFC_DATASET_ROUNDTRIP_VALUES_MISMATCH:'+label)


def _verify(dataset, episode, raw):
    row = dataset.episodes[episode.episode_id]
    shard = dataset.shards[row['shard']]
    with h5py.File(dataset._verified(shard['path'],shard['sha256']), 'r') as stored:
        group = stored[episode.episode_id]
        for name in ('spike_ticks','spike_unit_indices','control_ticks','control_values'):
            _exact(group[name], np.asarray(getattr(episode,name)), name)
        for name, values in episode.privileged.items():
            _exact(group['privileged'][name], values, 'privileged/'+name)
        metadata = json.loads(group.attrs['metadata_json'])
        if (metadata['split']!='development' or metadata['source_record']!=episode.source_record
                or canonical_json(metadata['control_features'])!=canonical_json(episode.control_features)):
            raise ValueError('PFC_DATASET_ROUNDTRIP_METADATA_MISMATCH')
    total = dataset.catalog['circuits'][episode.circuit_id]['units']
    bin_ticks = int(round(5.0/episode.dt_ms))
    if episode.duration_ticks % bin_ticks or episode.duration_ticks//bin_ticks < 2:
        raise ValueError('PFC_DATASET_VERIFICATION_CLOCK_UNSUPPORTED')
    remaining = episode.duration_ticks//bin_ticks
    start, events, values_checked = 0,0,0
    integral = np.zeros(len(episode.control_features),dtype=np.float64)
    while remaining:
        bins = min(VERIFY_WINDOW_BINS,remaining)
        if remaining-bins==1: bins-=1
        history = bins//2
        stop = start+bins*bin_ticks
        view = dataset.window(episode.episode_id,boundary_tick=start+history*bin_ticks,
            history_ticks=history*bin_ticks,horizon_ticks=(bins-history)*bin_ticks,bin_ticks=bin_ticks)
        if 'privileged' in view or any(name in view for name in episode.privileged):
            raise ValueError('PFC_DATASET_PRIVILEGED_STATE_EXPOSED_IN_CAUSAL_WINDOW')
        actual = np.concatenate((view['history'],view['targets']))
        left,right = np.searchsorted(raw['spike_ticks'],[start,stop],side='left')
        indices = ((raw['spike_ticks'][left:right]-start)//bin_ticks)*total+raw['spike_unit_indices'][left:right]
        expected = np.bincount(indices,minlength=bins*total).reshape(bins,total)
        if not np.array_equal(actual,expected):
            raise ValueError('PFC_DATASET_ALL_UNIT_WINDOW_COUNTS_MISMATCH')
        events+=int(actual.sum());values_checked+=int(actual.size)
        integral+=view['known_controls'].sum(axis=0)*bin_ticks
        start=stop;remaining-=bins
    lengths=np.diff(np.r_[episode.control_ticks,episode.duration_ticks])
    expected_integral=(episode.control_values*lengths[:,None]).sum(axis=0)
    if (events!=len(episode.spike_ticks) or not np.allclose(integral,expected_integral,
            rtol=READOUT_RTOL,atol=READOUT_ATOL)):
        raise ValueError('PFC_DATASET_EVENT_OR_CONTROL_INTEGRAL_MISMATCH')
    return dict(case_id=episode.episode_id,status='PASS',event_values=events,
                all_unit_bin_values_checked=values_checked,control_integral_natively_timed=True,
                privileged_states_exposed_by_default_window=False)



def _benchmark_registration(config):
    """Verify a compiled prospective episode without promoting its circuit alias."""
    from .benchmark_tasks import pfc_working_memory_config, task_protocol
    from .pfc_source_execution import _case_config

    candidate = config['benchmark_episode_candidate']
    identity = candidate['identity']
    expected = pfc_working_memory_config(config, config.get('benchmark_task_protocol',
        task_protocol('pfc_working_memory', pfc_dt_ms=config.get('dt_ms', 0.05))), identity)
    if expected != config:
        raise ValueError('PFC_DATASET_BENCHMARK_COMPILER_BINDING_MISMATCH')
    case_id = identity['episode_id']
    case, _, dimensions = _case_config(config, case_id)
    registered = {case_id: (case['external_noise_seed'], candidate['condition_id'])}
    return config, registered, [case['external_noise_seed']], dimensions


def export(source_runs, config_path, output, *, input_origins=None, distractor_baseline=None):
    config_path, output = Path(config_path),Path(output)
    if output.exists(): raise FileExistsError(output)
    candidate_config = yaml.safe_load(config_path.read_text())
    benchmark = isinstance(candidate_config, dict) and 'benchmark_episode_candidate' in candidate_config
    if benchmark:
        if distractor_baseline is not None:
            raise ValueError('PFC_DATASET_BENCHMARK_AND_HISTORICAL_DISTRACTOR_MODES_CONFLICT')
        config, registered, seeds, dimensions = _benchmark_registration(candidate_config)
    elif distractor_baseline is None:
        config,registered,seeds,dimensions = observed_source_registration(config_path)
    else:
        task = distractor_registration(config_path, distractor_baseline)
        config,registered,dimensions = task['candidate'],task['candidate_registry'],task['dimensions']
        seeds = [task['candidate']['cases'][0]['source_rng_seed']]
    config_sha=sha256_file(config_path)
    origins = None if input_origins is None else json.loads(Path(input_origins).read_text())
    if origins is not None:
        if not isinstance(origins,list):raise ValueError('PFC_DATASET_ORIGIN_TABLE_INVALID')
        mapped={str(Path(row['staged']).resolve()):row for row in origins}
        if len(mapped)!=len(origins):raise ValueError('PFC_DATASET_DUPLICATE_STAGED_ORIGIN')
    headers=[];seen=set()
    for path in source_runs:
        status,header=_source_header(path,config,registered,config_sha)
        if header is None or status['status']!='COMPLETE':
            raise ValueError('PFC_DATASET_COMPLETE_SOURCE_REQUIRED')
        if status['case_id'] in seen: raise ValueError('PFC_DATASET_DUPLICATE_SOURCE_CASE')
        if origins is not None:
            origin=mapped.get(str(header['archive'].resolve()))
            if (origin is None or origin['case_id']!=status['case_id']
                    or origin['delivery_sha256']!=status['delivery_sha256']):
                raise ValueError('PFC_DATASET_STAGED_ORIGIN_BINDING_MISMATCH')
            durable_archive=Path(origin['archive'])
            if (not (durable_archive/'delivery.json').is_file()
                    or sha256_file(durable_archive/'delivery.json')!=status['delivery_sha256']):
                raise ValueError('PFC_DATASET_DURABLE_ORIGIN_BINDING_MISMATCH')
            header['binding']['path']=str(durable_archive/'source_run')
        seen.add(status['case_id']);headers.append(header)
    if not headers: raise ValueError('PFC_DATASET_SOURCE_REQUIRED')
    if origins is not None and len(headers)!=len(origins):raise ValueError('PFC_DATASET_ORIGIN_DENOMINATOR_MISMATCH')
    output.mkdir(parents=True)
    started=time.perf_counter()
    report=dict(status='EXPORTING',started_utc=datetime.now(timezone.utc).isoformat(),
        source_paths=[str(Path(header['binding']['path']).parent) for header in headers],registered_source_cases=len(registered),
        selected_source_cases=len(headers),new_source_simulations=0,production_episodes=0,
        scientific_fidelity_qualified=False,source_episodes_added_to_campaign=0,verification=[])
    try:
        binding=dict(schema=SCHEMA,scope='source_reproduction',scenario_id='pfc_working_memory',
            source_revision=SOURCE_REVISION,source_entry_sha256=SOURCE_ENTRY_SHA256,
            sample_unit='complete_mechanistic_episode',config_sha256=config_sha,resolved_source_config=config,
            source_bindings=[dict(header['binding']) for header in headers],
            control_timing='observed_source_operation_segments',
            control_window_reduction='time mean of source assignment amplitudes and source schedule indicators',
            admissible_actions='registered cue/distractor source commands; per-unit currents are verification data',
            exporter_sha256=sha256_file(Path(__file__)))
        common=None
        with CircuitArchiveWriter(output/'dataset',binding,target_shard_bytes=1) as writer:
            for header in headers:
                raw,execution,paired=_source_values(header,config,registered,dimensions,config_sha)
                observation=header['binding']['applied_current_observation']
                if observation['status']!='VERIFIED_ACTUAL_CURRENT':
                    raise ValueError('PFC_DATASET_OBSERVED_CURRENT_REQUIRED')
                if common is None: common=execution
                if execution!=common: raise ValueError('PFC_DATASET_MIXED_SOURCE_EXECUTION')
                units=_units(config,header,dimensions);writer.register_circuit(units)
                case=header['report']['selected_case']
                observed=_arrays(_bound(header,'applied_current_observation.npz'))
                terminal=_arrays(_bound(header,'terminal_state.npz'))
                ticks,values,features=_controls(config,case,observed)
                private=dict(initial_voltage_mV=paired['initial'])
                private.update({'terminal__'+name:value for name,value in terminal.items()})
                private.update({'delivered_current__'+name:value for name,value in observed.items()})
                policy = header['report']['random_stream_policy']
                source_seeds = dict(header['report']['seeds'])
                if policy == 'source_global':
                    block_id = 'pfc_source_block_' + str(source_seeds['source_rng_seed'])
                    stream_semantics = 'source_global_rng_not_separated'
                else:
                    block_id = 'pfc_initial_' + str(source_seeds['initial_rng_seed']) + '_external_' + str(source_seeds['external_noise_seed'])
                    stream_semantics = 'independent_initial_voltage_and_external_noise_streams'
                record=dict(complete_unit_recording=True,simulation_completed=True,
                    recorded_until_tick=dimensions['duration_ticks'],expected_duration_ticks=dimensions['duration_ticks'],
                    core_unit_count=dimensions['coupled_core_units'],source_revision=SOURCE_REVISION,
                    native_source_sha256=SOURCE_ENTRY_SHA256,circuit_provenance=units.circuit_provenance,
                    source_delivery_sha256=header['binding']['delivery_sha256'],
                    source_report_sha256=header['binding']['report_sha256'],
                    raw_sha256=header['report']['outputs']['raw.npz']['sha256'],
                    final_rng_sha256=header['report']['outputs']['final_rng.npz']['sha256'],
                    initial_and_background_streams=stream_semantics,
                    connection_identity_kind='fixed_source_rule_and_parameters_not_native_edge_replay',
                    source_unit_metadata=json.loads(_bound(header, 'unit_metadata.json').read_text()),
                    applied_current_observation=observation,
                    privileged_roles=dict(initial_voltage='source provenance, not model input',
                        terminal='terminal microscopic states, not causal model input',
                        delivered_current='actual delivery verification; not independently admissible action channels'))
                if benchmark:
                    record['benchmark_episode_candidate'] = config['benchmark_episode_candidate']
                episode=CircuitEpisode(case['case_id'],units.circuit_id,block_id,
                    'development',dimensions['duration_ticks'],config['dt_ms'],raw['spike_ticks'],raw['spike_unit_indices'],
                    ticks,values,features,seeds=source_seeds,source_record=record,
                    privileged=private,task_outcome='not_scientifically_assessed')
                writer.add(episode)
                dataset=CircuitDataset(output/'dataset',splits=('development',))
                report['verification'].append(_verify(dataset,episode,raw))
        report.update(status='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET',complete_episodes=len(headers),
            missing_registered_cases=[case for case in registered if case not in seen],
            dataset_catalog_sha256=sha256_file(output/'dataset/dataset.json'),
            source_fidelity_qualified=False)
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        report['completed_utc']=datetime.now(timezone.utc).isoformat()
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,action='append',required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh development output directory')
    parser.add_argument('--input-origins',type=Path,help='Existing staged-input provenance from the source readout')
    parser.add_argument('--distractor-baseline',type=Path,
                        help='Completed baseline archive for the registered single-distractor protocol')
    args=parser.parse_args(argv)
    print(json.dumps(export(args.source_run,args.config,args.output,input_origins=args.input_origins,
                            distractor_baseline=args.distractor_baseline),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

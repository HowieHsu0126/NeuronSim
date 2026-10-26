"""Lossless source-observation archive for the repeated full V1 native case.

Uses the shared circuit archive and preserves source point-neuron metadata.
Incoming source spikes and boundary microscopic states are privileged evidence;
visual commands are optionally joined from an independently verified source archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits
from .population_field import PopulationFieldMapping
from .provenance import canonical_json, freeze_json, sha256_file
from .v1_nest_input_replay import bound_json
from .v1_configured_inputs import delivered_file, validate_plan
from .v1_lgn_spikes import input_plan_identity
from .v1_nest_repeatability import load_outputs
from .v1_nest_time_semantics import encode_precise_times
from .v1_visual_controls import load_visual_controls

PAPER_SHA256 = '70fc6fe5f57d1d258710dff5bce8295590385737106e103278e1fcd6e39a4289'
PAPER_DOI = '10.1126/sciadv.abq7592'


def exact_ticks(times_ms, dt_ms, duration_ticks):
    times = np.asarray(times_ms)
    if (times.ndim != 1 or times.dtype.kind not in 'iuf' or not np.isfinite(times).all()
            or not np.isfinite(dt_ms) or dt_ms <= 0 or type(duration_ticks) is not int or duration_ticks <= 0):
        raise ValueError('V1_OBSERVATION_CLOCK_INVALID')
    values = times / dt_ms
    if (np.any(values < 0) or np.any(values > duration_ticks) or np.any(values != np.floor(values))
            or not np.array_equal(values*dt_ms, times)):
        raise ValueError('V1_OBSERVATION_NONINTEGER_NATIVE_TIME')
    return values.astype(np.int64)


def coordinate_authority(directory, graph):
    source = json.loads((directory/'source.json').read_text())
    if (source['sha256'] != PAPER_SHA256 or sha256_file(directory/'paper.xml') != PAPER_SHA256
            or sha256_file(directory/'load_sparse.py') != source['author_loader_sha256']
            or source['author_loader_sha256'] != graph['author_loader_sha256']
            or source['coordinate_units'] != 'um' or source['hdf5_attribute_declaration_claimed'] is not False):
        raise ValueError('V1_COORDINATE_SOURCE_EVIDENCE_CHANGED')
    article = ET.parse(directory/'paper.xml')
    if PAPER_DOI not in [node.text for node in article.iter('article-id') if node.get('pub-id-type') == 'doi']:
        raise ValueError('V1_COORDINATE_PAPER_IDENTITY_CHANGED')
    return source


def metadata_units(nodes_path, types_path, native_identity, source_revision, provenance, semantics,
                   *, circuit_id='v1_original_full_cortex', allowed_locations=('Cortex',)):
    with types_path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter=' '))
    types = {int(row['node_type_id']): row for row in rows}
    if len(types) != len(rows):
        raise ValueError('V1_OBSERVATION_DUPLICATE_SOURCE_TYPE')
    with h5py.File(nodes_path, 'r') as source:
        group = source['nodes/v1']
        ids, model_ids, groups, indices = [group[name][:] for name in ('node_id','node_type_id','node_group_id','node_group_index')]
        n = len(ids)
        if (any(value.shape != (n,) or value.dtype.kind not in 'iu' for value in (ids,model_ids,groups,indices))
                or not np.array_equal(np.sort(ids),np.arange(n)) or not np.array_equal(np.sort(indices),np.arange(n))
                or not np.all(groups==0) or set(map(int,model_ids)) != set(types)):
            raise ValueError('V1_OBSERVATION_SOURCE_NODE_IDENTITY_INVALID')
        positions = np.column_stack([group['0/'+axis][:][indices] for axis in ('x','y','z')])
        tuning = group['0/tuning_angle'][:][indices] if 'tuning_angle' in group['0'] else None
        declared_units = {axis:group['0/'+axis].attrs.get('units') for axis in ('x','y','z')}
        for units in declared_units.values():
            if isinstance(units,bytes): units=units.decode('utf-8')
            if units is not None and units not in ('um','µm','μm','micron','microns','micrometer','micrometers'):
                raise ValueError('V1_COORDINATE_UNIT_ATTRIBUTE_CONFLICT')
    order = np.argsort(ids)
    ids, model_ids, positions = ids[order], model_ids[order], positions[order]
    tuning = tuning[order] if tuning is not None else None
    if (not np.array_equal(ids,native_identity['source_id'])
            or not np.array_equal(model_ids,native_identity['node_type_id']) or not np.isfinite(positions).all()):
        raise ValueError('V1_OBSERVATION_NATIVE_METADATA_JOIN_DIFFERS')
    for row in types.values():
        if row['population']!='v1' or row['location'] not in allowed_locations or row['model_type']!='point_process':
            raise ValueError('V1_OBSERVATION_NONCORTICAL_UNIT_METADATA')
    labels = np.asarray([types[int(identity)]['pop_name'] for identity in model_ids])
    mapping = PopulationFieldMapping.from_groups(ids,labels.tolist(),definition='Publisher node-type table pop_name; all source cortical point neurons.')
    units = CircuitUnits(circuit_id,ids,labels,mapping.group_index,tuple(mapping.field_ids),
                         semantics,source_revision,positions,'um',provenance,node_type_ids=model_ids,source_tuning_angle_deg=tuning)
    units.validate()
    return units, types


def prepare_spikes(activity, units, duration_ticks, dt_ms):
    ticks = exact_ticks(activity['spike_time_ms'],dt_ms,duration_ticks)
    ids = np.asarray(activity['spike_source_id'])
    if ids.shape != ticks.shape or ids.dtype.kind not in 'iu':
        raise ValueError('V1_OBSERVATION_SPIKE_IDENTITIES_INVALID')
    indices = np.searchsorted(units.unit_ids,ids)
    if np.any(indices>=len(units.unit_ids)) or not np.array_equal(units.unit_ids[indices],ids):
        raise ValueError('V1_OBSERVATION_UNKNOWN_SOURCE_SPIKE_UNIT')
    counts = np.bincount(indices,minlength=len(units.unit_ids))
    if not np.array_equal(counts,activity['spike_counts']):
        raise ValueError('V1_OBSERVATION_SPIKE_COUNT_READBACK_DIFFERS')
    order = np.lexsort((indices,ticks))
    return ticks[order],indices[order]


def verify_export(dataset, episode, units):
    row = dataset.episodes[episode.episode_id]; shard = dataset.shards[row['shard']]
    circuit = dataset.catalog['circuits'][units.circuit_id]
    with h5py.File(dataset._verified(circuit['path'],circuit['sha256']),'r') as source:
        for key,expected in [('unit_ids',units.unit_ids),('node_type_ids',units.node_type_ids),
                             ('coordinates',units.coordinates),('field_index',units.field_index)]:
            if not np.array_equal(source[key][:],expected):raise ValueError('V1_STATIC_METADATA_ROUNDTRIP_DIFFERS:'+key)
        tuning = units.source_tuning_angle_deg
        if ('source_tuning_angle_deg' in source) != (tuning is not None) or (tuning is not None and not np.array_equal(source['source_tuning_angle_deg'][:], tuning)):
            raise ValueError('V1_STATIC_METADATA_ROUNDTRIP_DIFFERS:source_tuning_angle_deg')
        if not np.array_equal(source['populations'].asstr()[:],units.populations):
            raise ValueError('V1_SOURCE_POPULATION_ROUNDTRIP_DIFFERS')
    with h5py.File(dataset._verified(shard['path'],shard['sha256']),'r') as source:
        stored = source[episode.episode_id]
        for key in ('spike_ticks','spike_unit_indices','control_ticks','control_values'):
            if not np.array_equal(stored[key][:],getattr(episode,key)):
                raise ValueError('V1_EPISODE_ROUNDTRIP_DIFFERS:'+key)
        for key,value in episode.privileged.items():
            if not np.array_equal(stored['privileged'][key][:],value):
                raise ValueError('V1_PRIVILEGED_EVIDENCE_ROUNDTRIP_DIFFERS:'+key)
    windows=[]
    boundaries = [4, episode.duration_ticks-4]
    ordinary_ticks = episode.spike_ticks[episode.spike_ticks < episode.duration_ticks]
    if len(ordinary_ticks):
        # Silent boundaries alone cannot verify a real count/field observation window.
        active_tick = int(ordinary_ticks[len(ordinary_ticks)//2])
        boundaries.append(max(4, min(episode.duration_ticks-4, active_tick)))
    if episode.control_features:
        # Include each contrast transition; source phase updates retain their full raw schedule.
        changes = episode.control_ticks[1:][np.diff(episode.control_values[:, 0]) != 0]
        boundaries += [int(tick) for tick in changes if 4 <= tick <= episode.duration_ticks-4]
    for boundary in dict.fromkeys(boundaries):
        batch = dataset.window(episode.episode_id,boundary_tick=boundary,history_ticks=4,horizon_ticks=4,bin_ticks=1)
        start,stop=boundary-4,boundary+4
        selected=(episode.spike_ticks>=start)&(episode.spike_ticks<stop)
        expected=np.zeros((8,len(units.unit_ids)),dtype=np.int64)
        np.add.at(expected,(episode.spike_ticks[selected]-start,episode.spike_unit_indices[selected]),1)
        actual=np.concatenate((batch['history'],batch['targets']))
        fields=np.zeros((8,len(units.field_names)),dtype=np.int64)
        np.add.at(fields,(episode.spike_ticks[selected]-start,units.field_index[episode.spike_unit_indices[selected]]),1)
        expected_controls = np.zeros((8, len(episode.control_features)))
        for sample, tick in enumerate(range(start, stop)):
            control_index = np.searchsorted(episode.control_ticks, tick, side='right')-1
            if control_index >= 0:
                expected_controls[sample] = episode.control_values[control_index]
        if (not np.array_equal(actual,expected)
                or not np.array_equal(np.concatenate((batch['history_fields'],batch['target_fields'])),fields)
                or not np.array_equal(batch['known_controls'], expected_controls)
                or batch['control_features'] != list(episode.control_features)
                or not batch['observation_mask'].all()
                or 'privileged' in batch or not np.array_equal(batch['unit_ids'],units.unit_ids)):
            raise ValueError('V1_SHARED_WINDOW_READBACK_DIFFERS')
        windows.append({'start_tick':start,'stop_tick':stop,'all_units':len(units.unit_ids),'events':int(selected.sum())})
    return {'raw_events_exact':len(episode.spike_ticks),'static_unit_metadata_exact':len(units.unit_ids),
            'endpoint_events_retained':int(np.count_nonzero(episode.spike_ticks==episode.duration_ticks)),
            'silent_units_retained':int(np.count_nonzero(np.bincount(episode.spike_unit_indices,minlength=len(units.unit_ids))==0)),
            'windows':windows,'window_privileged_fields_exposed':False,'admissible_control_features':len(episode.control_features)}


def input_case_metadata(native, circuit_id):
    """Keep generated input identities separate from the historical replay case."""
    seeds = {'native_kernel_rng_seed': native['kernel_rng_seed']}
    generated = native.get('generated_lgn')
    if generated is None:
        return 'v1_original_input_reproduction', 'v1_original_input_block', seeds, {}
    input_digest, background_digest = generated['input_identity_sha256'], generated['original_plan_sha256']
    for value in (input_digest, background_digest):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('V1_GENERATED_INPUT_CASE_DIGEST_INVALID')
    settings = generated['config']['spike_generation']
    seeds['lgn_starting_noise_seed'] = settings['starting_noise_seed']
    record = {key: generated[key] for key in ('spike_report_sha256', 'input_identity_sha256', 'trial', 'original_plan_sha256',
                                            'input_origin_sha256', 'config')}
    record['admissible_control_mapping_qualified'] = False
    semantic_case = dict(circuit_id=circuit_id, stimulus=generated['config']['stimulus'],
                         starting_noise_seed=settings['starting_noise_seed'], trial=generated['trial'],
                         input_identity=input_digest, background_identity=background_digest)
    case_digest = hashlib.sha256(canonical_json(semantic_case).encode('utf-8')).hexdigest()
    return ('v1_generated_lgn_'+case_digest,
            'v1_lgn_start_'+str(settings['starting_noise_seed'])+'_background_'+background_digest,
            seeds, record)


def export(native_archive, repeatability_archive, input_replay_archive, asset_root, output,
           visual_controls_archive=None):
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report={'status':'FAILED_V1_SOURCE_OBSERVATION_EXPORT','started_utc':datetime.now(timezone.utc).isoformat(),
            'new_source_simulations':0,'production_episodes':0,'source_fidelity_qualified':False,
            'canonical_observation_map_frozen':False,'admissible_visual_control_mapping_qualified':False}
    try:
        native,delivery=bound_json(native_archive,'network/report.json')
        repeated,_=bound_json(repeatability_archive,'repeatability.json')
        graph,_=bound_json(native_archive,'graph/graph_identity.json')
        if (delivery['status']!='PASS' or delivery['exit_code']!=0
                or native['status']!='NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE'
                or repeated['status']!='PASS_NATIVE_V1_REPEATABILITY'
                or repeated['first_report_sha256']!=sha256_file(native_archive/'network/report.json')
                or not all(repeated['checks'].values())):
            raise ValueError('V1_REPEATED_COMPLETE_NATIVE_SOURCE_REQUIRED')
        data=load_outputs(native_archive/'network',native)
        nodes=asset_root/'node_metadata_complete/v1_nodes.h5';types=asset_root/'node_types/source_asset.txt'
        validated=json.loads((asset_root/'node_validation.json').read_text())
        if (sha256_file(asset_root/'node_validation.json')!=graph['reference_metadata_sha256']
                or validated['status']!='NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS'
                or sha256_file(nodes)!=graph['member_sha256']['GLIF_network2/network/v1_nodes.h5']
                or sha256_file(types)!=validated['node_types_sha256']):
            raise ValueError('V1_SOURCE_STATIC_METADATA_CHANGED')
        authority=coordinate_authority(asset_root/'original_graph_route/coordinate_semantics',graph)
        source_revision='source-graph-sha256:'+graph['member_sha256']['GLIF_network2/network_dat.pkl']
        parameters,_=bound_json(native_archive,'parameters/comparison.json')
        provenance={'connectivity_identity':graph['member_sha256']['GLIF_network2/network_dat.pkl'],
                    'parameter_identity':parameters['parameter_reference_sha256'],
                    'unit_metadata_identity':sha256_file(nodes),'node_types_identity':sha256_file(types),
                    'coordinate_unit_evidence':sha256_file(asset_root/'original_graph_route/coordinate_semantics/source.json')}
        units,type_table=metadata_units(nodes,types,data['cortical_identity.npz'],source_revision,provenance,authority['coordinate_semantics'])
        if len(units.unit_ids)!=native['coupled_cortical_neurons']:raise ValueError('V1_COMPLETE_UNIT_DENOMINATOR_DIFFERS')
        input_identity,_=bound_json(native_archive,'inputs/input_identity.json')
        clock=input_identity['source_config_clock']
        if clock!={'duration':3000.0,'dt':.25}:raise ValueError('V1_SOURCE_CLOCK_CHANGED')
        dt=clock['dt'];duration=int(clock['duration']/dt)
        if native['native_duration_ms']!=dt*duration:raise ValueError('V1_SOURCE_DURATION_CHANGED')
        ticks,indices=prepare_spikes(data['cortical_activity.npz'],units,duration,dt)
        if native.get('generated_lgn') is not None:
            input_path = native_archive/'network/source_inputs.npz'
            if sha256_file(input_path) != native['artifacts']['source_inputs.npz']:
                raise ValueError('V1_GENERATED_INPUT_EVIDENCE_CHANGED')
        else:
            input_report,_=bound_json(input_replay_archive,'native_replay/report.json')
            input_path=input_replay_archive/'native_replay/source_inputs.npz'
            if (sha256_file(input_path)!=input_report['artifacts']['source_inputs.npz']
                    or input_report['source_input_member_sha256']!=input_identity['input_member_sha256']):
                raise ValueError('V1_INPUT_EVIDENCE_CHANGED')
        with np.load(input_path,allow_pickle=False) as source:incoming={key:source[key] for key in source.files}
        full_inputs=data['input_identity.npz']
        if (not np.array_equal(incoming['population'],full_inputs['population'])
                or not np.array_equal(incoming['source_id'],full_inputs['source_id'])
                or not np.array_equal(incoming['source_counts'],full_inputs['source_event_counts'])
                or len(incoming['source_event_time_ms'])!=native['input_events']):
            raise ValueError('V1_SOURCE_INPUT_POPULATION_JOIN_DIFFERS')
        timing,_=bound_json(native_archive,'timing/diagnosis/report.json')
        encoded=encode_precise_times(incoming['source_event_time_ms'],timing['native_witness']['tics_per_ms'])
        endpoint=ticks==duration
        private={'initial_vm_mV':data['cortical_identity.npz']['initial_vm_mV'],
                 'terminal_vm_mV':data['cortical_activity.npz']['final_vm_mV'],
                 'final_endpoint_unit_indices':indices[endpoint],
                 'final_endpoint_ticks':ticks[endpoint],
                 'incoming_population_index':incoming['population'],'incoming_source_ids':incoming['source_id'],
                 'incoming_event_source_indices':incoming['source_event_unit'],
                 'incoming_raw_event_time_ms':incoming['source_event_time_ms'],
                 'incoming_native_event_steps':encoded['steps'],'incoming_native_event_offsets_ms':encoded['offsets_ms']}
        source_record={'complete_unit_recording':True,'simulation_completed':True,
            'recorded_until_tick':duration,'expected_duration_ticks':duration,'core_unit_count':len(units.unit_ids),
            'source_revision':source_revision,'circuit_provenance':provenance,
            'native_report_sha256':sha256_file(native_archive/'network/report.json'),
            'repeatability_report_sha256':sha256_file(repeatability_archive/'repeatability.json'),
            'ordinary_events':int((~endpoint).sum()),'final_endpoint_events':int(endpoint.sum()),
            'endpoint_semantics':'Endpoint events remain in the raw total and have a separate privileged view; half-open windows exclude them.',
            'privileged_roles':{'incoming':'Source spike delivery evidence, not qualified visual stimulus actions.',
                                'boundary_state':'Microscopic source state; not causal model input.'},
            'coordinate_unit_evidence_basis':authority['unit_evidence_basis'],
            'input_population_labels_qualified':False}
        case_id, block_id, seeds, generated_record = input_case_metadata(native, units.circuit_id)
        if generated_record:
            source_record['generated_lgn'] = generated_record
        controls = dict(control_ticks=np.empty(0, dtype=np.int64),
                        control_values=np.empty((0, 0)), control_features=())
        if visual_controls_archive is not None:
            controls = load_visual_controls(visual_controls_archive, native)
            source_record['visual_commands'] = controls.pop('source_control_record')
        episode=CircuitEpisode(case_id,units.circuit_id,block_id,'development',
            duration,dt,ticks,indices,controls['control_ticks'],controls['control_values'],controls['control_features'],
            seeds=seeds,source_record=source_record,
            privileged=private,task_outcome='not_scientifically_assessed')
        # Both cases reproduce declared source inputs; generated recipe identity remains in source_record.
        binding={'schema':SCHEMA,'scope':'source_reproduction','scenario_id':'v1_visual','source_revision':source_revision,
                 'sample_unit':'complete_mechanistic_episode','native_implementation_sha256':native['implementation_sha256'],
                 'source_bindings':[{'job_id':delivery['job_id'],'delivery_sha256':sha256_file(native_archive/'delivery.json')}],
                 'admissible_actions':('verified declared source-frame commands for this development case'
                                       if visual_controls_archive is not None else 'not qualified; no control features exported'),
                 'coordinate_unit_evidence_basis':authority['unit_evidence_basis'],'exporter_sha256':sha256_file(Path(__file__))}
        with CircuitArchiveWriter(output/'dataset',binding) as writer:
            writer.register_circuit(units);writer.add(episode)
        dataset=CircuitDataset(output/'dataset',splits=('development',))
        verification=verify_export(dataset,episode,units)
        populations=dict(Counter(units.populations.tolist()))
        totals=np.zeros(len(units.field_names),dtype=np.int64)
        np.add.at(totals,units.field_index,data['cortical_activity.npz']['spike_counts'])
        field_units=np.bincount(units.field_index,minlength=len(units.field_names))
        report.update(status='EXPORTED_VERIFIED_V1_SOURCE_OBSERVATION',complete_source_cases=1,independent_source_cases=1,
            native_job_id=delivery['job_id'],repeatability_job_id=repeated['second_job_id'],
            source_model_types=len(type_table),source_populations=populations,coordinate_units='um',
            coordinate_unit_evidence_basis=authority['unit_evidence_basis'],verification=verification,
            population_readout=[{'population':str(name),'units':int(n),'all_events':int(count),
                                 'mean_rate_hz_including_endpoint':float(count/n/3.)}
                                for name,n,count in zip(units.field_names,field_units,totals)],
            exported_declared_visual_controls=visual_controls_archive is not None,
            source_visual_command_record=source_record.get('visual_commands'),
            dataset_catalog_sha256=sha256_file(output/'dataset/dataset.json'))
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/'report.json',report)
    return report


POINTNET_ARRAYS = ('native/cortical_state.npz', 'native/cortical_spikes.npz', 'native/input_events.npz')
POINTNET_STATUSES = {'diagnostic_pulses': 'POINTNET_NATIVE_DIAGNOSTIC_COMPLETE',
                    'constructed_visual_inputs': 'POINTNET_NATIVE_VISUAL_INPUTS_COMPLETE_DEVELOPMENT'}
POINTNET_DELIVERIES = {'diagnostic_pulses': 'PASS_POINTNET_NATIVE_DIAGNOSTIC',
                      'constructed_visual_inputs': 'PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT'}
COORDINATE_SOURCES = ('original_node_parameters', 'original_node_functions', 'original_build_network')
OFFICIAL_MODEL_URL = 'https://brain-map.org/our-research/computational-modeling/models-of-the-mouse-primary-visual-cortex'


def _content_digest(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def read_pointnet_outputs(archive):
    """Verify the two process receipts and the arrays this exporter consumes."""
    native, delivery = bound_json(archive, 'primary/report.json')
    replay, _ = bound_json(archive, 'replay/report.json')
    comparison, _ = bound_json(archive, 'comparison/report.json')
    mode = native.get('input_mode')
    if (mode not in POINTNET_STATUSES or delivery.get('status') != POINTNET_DELIVERIES[mode]
            or delivery.get('exit_code') != 0
            or any(row.get('status') != POINTNET_STATUSES[mode] or row.get('exit_code') != 0
                   or row.get('input_mode') != mode for row in (native, replay))
            or comparison.get('status') != 'POINTNET_NATIVE_REPLAY_EQUAL' or comparison.get('exit_code') != 0
            or comparison.get('all_native_outputs_equal') is not True
            or comparison.get('independent_processes') is not True
            or comparison.get('binding_or_runtime_differences') != []
            or comparison.get('left_report_sha256') != sha256_file(archive/'primary/report.json')
            or comparison.get('right_report_sha256') != sha256_file(archive/'replay/report.json')):
        raise ValueError('V1_POINTNET_COMPLETE_BOUND_REPLAY_REQUIRED')
    for key in ('bindings', 'mode', 'input_event_schema', 'array_artifacts', 'kernel_rng_seed',
                'simulated_ms', 'cortical_units', 'source_denominators', 'connections',
                'implementation_sha256', 'nest_version', 'local_num_threads', 'input_readback'):
        if native[key] != replay[key]:
            raise ValueError('V1_POINTNET_REPLAY_CONTEXT_DIFFERS:' + key)
    arrays = native['array_artifacts']
    expected = set(arrays) | {'native/parameters.json'}
    compared = [row['path'] for row in comparison['files']]
    if (len(set(arrays)) != len(arrays) or not set(POINTNET_ARRAYS).issubset(arrays)
            or len(set(compared)) != len(compared) or set(compared) != expected
            or any(row.get('equal') is not True for row in comparison['files'])):
        raise ValueError('V1_POINTNET_REQUIRED_OUTPUT_COMPARISON_MISSING')
    result = {}
    for relative in POINTNET_ARRAYS:
        first, second = [delivered_file(archive, prefix+'/'+relative) for prefix in ('primary', 'replay')]
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError('V1_POINTNET_PAIRED_ARRAYS_DIFFER:' + relative)
            result[Path(relative).name] = {key: a[key] for key in a.files}
    paths = [delivered_file(archive, prefix+'/native/parameters.json') for prefix in ('primary', 'replay')]
    parameters = json.loads(paths[0].read_text())
    if parameters != json.loads(paths[1].read_text()):
        raise ValueError('V1_POINTNET_PAIRED_PARAMETERS_DIFFER')
    if (not parameters or any(row.get('all_units_exact') is not True or type(row.get('units')) is not int
                             or row['units'] < 1 for row in parameters.values())
            or sum(row['units'] for row in parameters.values()) != native['cortical_units']
            or native.get('native_parameters_and_initial_state_exact') is not True):
        raise ValueError('V1_POINTNET_ALL_UNIT_PARAMETER_READBACK_REQUIRED')
    return native, delivery, comparison, result, parameters


def constructed_coordinate_authority(archive, native):
    authority, delivery = bound_json(archive, 'authority.json')
    if (delivery.get('exit_code') != 0 or authority.get('exit_code') != 0
            or authority.get('status') != 'CONSTRUCTED_V1_LENGTH_UNIT_SUPPORTED'
            or authority.get('coordinate_units') != 'um'
            or authority.get('glif_coordinate_preservation_required') is not True
            or authority.get('hdf5_unit_attribute_declaration_claimed') is not False
            or not authority.get('coordinate_semantics') or not authority.get('unit_evidence_basis')):
        raise ValueError('V1_CONSTRUCTED_COORDINATE_AUTHORITY_REQUIRED')
    if set(authority['source_bindings']) != set(COORDINATE_SOURCES):
        raise ValueError('V1_CONSTRUCTED_COORDINATE_SOURCE_BINDINGS_MISSING')
    parameters = None
    for role in COORDINATE_SOURCES:
        receipt_path = delivered_file(archive, 'sources/'+role+'/source.json')
        asset_path = delivered_file(archive, 'sources/'+role+'/source_asset.txt')
        receipt = json.loads(receipt_path.read_text())
        observed = dict(bytes=asset_path.stat().st_size, sha256=sha256_file(asset_path),
                        receipt_sha256=sha256_file(receipt_path))
        if (observed != authority['source_bindings'][role] or observed != native['bindings']['sources'][role]
                or receipt.get('artifact') != asset_path.name or receipt.get('bytes') != observed['bytes']
                or receipt.get('sha256') != observed['sha256']):
            raise ValueError('V1_CONSTRUCTED_COORDINATE_SOURCE_CHANGED:' + role)
        if role == 'original_node_parameters':
            parameters = json.loads(asset_path.read_text())
    if (authority['inner_radial_range'] != [1.,400.] or authority['outer_radial_range'] != [400.,845.]
            or parameters['inner_radial_range'] != authority['inner_radial_range']
            or parameters['outer_radial_range'] != authority['outer_radial_range']
            or set(parameters['locations']) != set(authority['cortical_locations'])
            or len(set(authority['cortical_locations'])) != len(authority['cortical_locations'])):
        raise ValueError('V1_CONSTRUCTED_COORDINATE_DOMAIN_DIFFERS')
    official = authority['official_page']
    page = delivered_file(archive, 'model_page.html')
    text = delivered_file(archive, 'model_page.txt')
    readme = delivered_file(archive, 'model_readme.txt')
    if (official['url'] != OFFICIAL_MODEL_URL or official['path'] != page.name
            or official['bytes'] != page.stat().st_size or official['sha256'] != sha256_file(page)
            or authority['official_text_sha256'] != sha256_file(text)
            or authority.get('model_readme_sha256', sha256_file(readme)) != sha256_file(readme)):
        raise ValueError('V1_CONSTRUCTED_OFFICIAL_COORDINATE_EVIDENCE_CHANGED')
    return authority


def verify_coordinate_preservation(units, biophysical_nodes):
    """Compare all coordinates by source ID; GLIF model assignments may differ."""
    with h5py.File(biophysical_nodes, 'r') as source:
        group = source['nodes/v1']
        ids, groups, indices = [group[key][:] for key in ('node_id','node_group_id','node_group_index')]
        n = len(units.unit_ids)
        if (any(value.shape != (n,) or value.dtype.kind not in 'iu' for value in (ids,groups,indices))
                or not np.array_equal(np.sort(ids),units.unit_ids) or not np.all(groups == 0)
                or not np.array_equal(np.sort(indices),np.arange(n))):
            raise ValueError('V1_BIOPHYSICAL_COORDINATE_IDENTITY_DIFFERS')
        for axis in ('x','y','z'):
            declaration = group['0/'+axis].attrs.get('units')
            if isinstance(declaration,bytes): declaration=declaration.decode('utf-8')
            if declaration is not None and declaration not in ('um','µm','μm','micron','microns','micrometer','micrometers'):
                raise ValueError('V1_BIOPHYSICAL_COORDINATE_UNIT_CONFLICT')
        positions = np.column_stack([group['0/'+axis][:][indices] for axis in ('x','y','z')])[np.argsort(ids)]
    if not np.array_equal(positions,units.coordinates):
        raise ValueError('V1_GLIF_BIOPHYSICAL_COORDINATES_DIFFER')
    return dict(source_ids_verified=n, coordinate_values_exact=n*3,
                comparison='Exact ID-aligned source values; no scaling or fitted tolerance.')


def pointnet_circuit_identity(native, network_identities, parameters):
    source_revision = 'pointnet-source-' + _content_digest(dict(
        source_hashes={key:value['sha256'] for key,value in native['bindings']['sources'].items()},
        bmtk_wheel_sha256=native['wheel']['sha256'], nest_version=native['nest_version']))
    parameter_identity = _content_digest({key:dict(model=row['model'],parameters=row['parameters'])
                                          for key,row in parameters.items()})
    connectivity_identity = _content_digest(network_identities)
    circuit_id = 'v1-constructed-' + _content_digest(dict(source_revision=source_revision,
        connectivity_identity=connectivity_identity,parameter_identity=parameter_identity))
    return circuit_id, source_revision, dict(connectivity_identity=connectivity_identity,
                                            parameter_identity=parameter_identity)


def pointnet_input_evidence(native, incoming):
    """Normalize only the two native schemas; keep incoming events privileged."""
    labels = np.asarray(incoming['population'])
    n = len(incoming['source_id'])
    if labels.shape != (n,) or labels.dtype.kind not in 'US' or set(labels.tolist()) != {'lgn','bkg'}:
        raise ValueError('V1_POINTNET_INPUT_POPULATION_LABELS_INVALID')
    population = np.asarray([0 if value == 'lgn' else 1 for value in labels],dtype=np.int64)
    integer_arrays = {}
    for key in ('source_id','native_id','inverse_source_id','source_event_counts'):
        value = np.asarray(incoming[key])
        if (value.shape != (n,) or value.dtype.kind not in 'iu' or np.any(value < 0)
                or (len(value) and int(value.max()) >= 2**63)):
            raise ValueError('V1_POINTNET_INPUT_INTEGER_INVENTORY_INVALID:' + key)
        integer_arrays[key] = value.astype(np.int64,copy=False)
    ids, counts = integer_arrays['source_id'], integer_arrays['source_event_counts']
    if (not np.array_equal(ids,integer_arrays['inverse_source_id'])
            or len(np.unique(integer_arrays['native_id'])) != n
            or incoming['precise_times'].shape != (n,) or incoming['precise_times'].dtype.kind != 'b'
            or not incoming['precise_times'].all()):
        raise ValueError('V1_POINTNET_INPUT_NATIVE_IDENTITY_DIFFERS')
    schema = native['input_event_schema']
    arrays = {key:np.asarray(incoming[key]) for key in
              ('requested_times_ms','native_times_ms','encoded_steps','encoded_offsets_ms')}
    if schema == 'diagnostic_pulse_matrix_v1' and native['input_mode'] == 'diagnostic_pulses':
        if not np.all(counts == 2) or any(value.shape != (n,2) for value in arrays.values()):
            raise ValueError('V1_POINTNET_DIAGNOSTIC_INPUT_SHAPE_DIFFERS')
        event_units = np.repeat(np.arange(n,dtype=np.int64),counts)
        arrays = {key:value.reshape(-1) for key,value in arrays.items()}
    elif schema == 'packed_source_events_v1' and native['input_mode'] == 'constructed_visual_inputs':
        event_units = np.asarray(incoming['source_event_unit'])
        if event_units.dtype.kind not in 'iu' or event_units.ndim != 1 or any(value.shape != event_units.shape for value in arrays.values()):
            raise ValueError('V1_POINTNET_PACKED_INPUT_SHAPE_DIFFERS')
    else:
        raise ValueError('V1_POINTNET_INPUT_SCHEMA_UNSUPPORTED')
    plan = dict(population=population,source_id=ids,source_counts=counts,
                source_event_unit=event_units,source_event_time_ms=arrays['requested_times_ms'])
    validate_plan(plan)
    readback = native['input_readback']
    flag = 'all_generator_pulses_exact' if native['input_mode'] == 'diagnostic_pulses' else 'all_generator_events_exact'
    if (readback['generators'] != n or readback['events'] != len(event_units)
            or readback.get(flag) is not True or readback.get('all_population_and_source_ids_exact') is not True):
        raise ValueError('V1_POINTNET_INPUT_READBACK_DENOMINATOR_DIFFERS')
    encoded = encode_precise_times(arrays['requested_times_ms'],readback['tics_per_ms'])
    if (not np.array_equal(encoded['steps'],arrays['encoded_steps'])
            or not np.array_equal(encoded['offsets_ms'],arrays['encoded_offsets_ms'])
            or not np.array_equal(encoded['decoded_times_ms'],arrays['native_times_ms'])):
        raise ValueError('V1_POINTNET_INPUT_PRECISE_TIME_READBACK_DIFFERS')
    identity = input_plan_identity(plan)
    if (native['input_mode'] == 'constructed_visual_inputs'
            and identity != native['bindings']['spike_inputs']['input_identity_sha256']):
        raise ValueError('V1_POINTNET_SOURCE_INPUT_CONTENT_IDENTITY_DIFFERS')
    private = dict(incoming_population_index=population,incoming_source_ids=ids,
        incoming_native_ids=integer_arrays['native_id'],incoming_source_event_counts=counts,
        incoming_event_source_indices=event_units,incoming_raw_event_time_ms=arrays['requested_times_ms'],
        incoming_native_event_time_ms=arrays['native_times_ms'],incoming_native_event_steps=arrays['encoded_steps'],
        incoming_native_event_offsets_ms=arrays['encoded_offsets_ms'],incoming_precise_times=incoming['precise_times'])
    return identity, private


def pointnet_case_metadata(native, circuit_id, initial_vm, input_identity, clock):
    values = np.asarray(initial_vm)
    if values.ndim != 1 or values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('V1_POINTNET_INITIAL_STATE_INVALID')
    initial_identity = hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()
    seeds = dict(native_kernel_rng_seed=native['kernel_rng_seed'])
    if native['input_mode'] == 'constructed_visual_inputs':
        config = native['bindings']['spike_inputs']['config']
        seeds.update(lgn_starting_noise_seed=config['spike_generation']['starting_noise_seed'],
                     background_starting_noise_seed=config['background']['starting_noise_seed'])
    if any(type(value) is not int for value in seeds.values()):
        raise ValueError('V1_POINTNET_DECLARED_INTEGER_RANDOM_STREAMS_REQUIRED')
    block = dict(circuit_id=circuit_id,initial_vm_identity=initial_identity,seeds=seeds)
    case = dict(block,input_mode=native['input_mode'],input_identity=input_identity,clock=clock)
    return ('v1-pointnet-'+_content_digest(case),'v1-pointnet-block-'+_content_digest(block),seeds,initial_identity)


def export_pointnet(native_archive, coordinate_authority_archive, output, visual_controls_archive=None):
    native_archive,coordinate_authority_archive,output = map(Path,(native_archive,coordinate_authority_archive,output))
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(status='FAILED_POINTNET_SOURCE_OBSERVATION_EXPORT',exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),new_source_simulations=0,production_episodes=0,
        scientific_qualification=False,source_fidelity_qualified=False,canonical_observation_map_frozen=False,
        admissible_visual_control_mapping_qualified=False)
    try:
        native,delivery,repeated,data,parameters=read_pointnet_outputs(native_archive)
        if visual_controls_archive is not None and native['input_mode'] != 'constructed_visual_inputs':
            raise ValueError('V1_POINTNET_VISUAL_CONTROLS_REQUIRE_CONSTRUCTED_INPUTS')
        authority=constructed_coordinate_authority(coordinate_authority_archive,native)
        network_identities={}
        for population in ('v1','lgn','bkg'):
            for suffix in ('nodes.h5','node_types.csv','v1_edges.h5','v1_edge_types.csv'):
                name=population+'_'+suffix
                upstream=native['bindings']['recurrent' if population=='v1' else 'inputs']['files']['primary/glif/'+name]
                # The passed paired native reports already bind unused graph files.
                # Only the V1 geometry/type files are consumed again by this exporter.
                if population=='v1' and suffix in ('nodes.h5','node_types.csv'):
                    path=delivered_file(native_archive,'primary/network/'+name)
                    actual=dict(bytes=path.stat().st_size,sha256=sha256_file(path))
                    if actual != upstream:raise ValueError('V1_POINTNET_REALIZED_NETWORK_FILE_CHANGED:'+name)
                network_identities[name]=dict(upstream)
        circuit_id,revision,provenance=pointnet_circuit_identity(native,network_identities,parameters)
        provenance.update(unit_metadata_identity=network_identities['v1_nodes.h5']['sha256'],
            node_types_identity=network_identities['v1_node_types.csv']['sha256'],
            coordinate_unit_evidence=sha256_file(coordinate_authority_archive/'authority.json'))
        state,spikes,incoming=[data[name] for name in ('cortical_state.npz','cortical_spikes.npz','input_events.npz')]
        units,type_table=metadata_units(native_archive/'primary/network/v1_nodes.h5',native_archive/'primary/network/v1_node_types.csv',
            state,revision,provenance,authority['coordinate_semantics'],circuit_id=circuit_id,allowed_locations=tuple(authority['cortical_locations']))
        bio=delivered_file(native_archive,'recurrent/primary/biophysical/v1_nodes.h5')
        expected=native['bindings']['recurrent']['files']['primary/biophysical/v1_nodes.h5']
        if bio.stat().st_size!=expected['bytes'] or sha256_file(bio)!=expected['sha256']:
            raise ValueError('V1_POINTNET_BIOPHYSICAL_COORDINATE_SOURCE_CHANGED')
        coordinate_check=verify_coordinate_preservation(units,bio)
        n=len(units.unit_ids)
        if (n!=native['cortical_units'] or not np.array_equal(state['source_id'],state['inverse_source_id'])
                or state['population'].shape!=(n,) or not np.all(state['population']=='v1')
                or state['native_id'].shape!=(n,) or state['native_id'].dtype.kind not in 'iu'
                or np.any(state['native_id']<=0) or len(np.unique(state['native_id']))!=n
                or any(state[key].shape!=(n,) or state[key].dtype.kind!='f' or not np.isfinite(state[key]).all()
                       for key in ('initial_vm_mV','terminal_vm_mV'))):
            raise ValueError('V1_POINTNET_CORTICAL_STATE_IDENTITY_DIFFERS')
        clock=native['bindings']['point_config']['source_clock']
        if clock!=dict(duration_ms=3000.,resolution_ms=.25) or native['simulated_ms']!=clock['duration_ms']:
            raise ValueError('V1_POINTNET_COMPLETED_SOURCE_CLOCK_DIFFERS')
        dt=clock['resolution_ms'];duration=int(clock['duration_ms']/dt)
        activity=dict(spike_time_ms=spikes['time_ms'],spike_source_id=spikes['source_id'],spike_counts=state['spike_count'])
        ticks,indices=prepare_spikes(activity,units,duration,dt)
        event_positions=np.searchsorted(units.unit_ids,spikes['source_id'])
        if (len(ticks)!=native['cortical_spike_events']
                or not np.array_equal(state['native_id'][event_positions],spikes['native_id'])):
            raise ValueError('V1_POINTNET_CORTICAL_EVENT_READBACK_DIFFERS')
        input_identity,private=pointnet_input_evidence(native,incoming)
        endpoint=ticks==duration
        private.update(initial_vm_mV=state['initial_vm_mV'],terminal_vm_mV=state['terminal_vm_mV'],
            final_endpoint_unit_indices=indices[endpoint],final_endpoint_ticks=ticks[endpoint])
        case_id,block_id,seeds,initial_identity=pointnet_case_metadata(native,circuit_id,state['initial_vm_mV'],input_identity,clock)
        source_record=dict(complete_unit_recording=True,simulation_completed=True,recorded_until_tick=duration,
            expected_duration_ticks=duration,core_unit_count=n,source_revision=revision,circuit_provenance=provenance,
            native_report_sha256=sha256_file(native_archive/'primary/report.json'),
            replay_report_sha256=sha256_file(native_archive/'replay/report.json'),
            repeatability_report_sha256=sha256_file(native_archive/'comparison/report.json'),
            coordinate_authority_sha256=sha256_file(coordinate_authority_archive/'authority.json'),
            coordinate_unit_evidence_basis=authority['unit_evidence_basis'],coordinate_preservation=coordinate_check,
            input_mode=native['input_mode'],input_event_schema=native['input_event_schema'],input_identity_sha256=input_identity,
            initial_vm_identity=initial_identity,incoming_population_names={'0':'lgn','1':'bkg'},
            random_stream_semantics='Native kernel and declared LGN/background sampler streams; not training seeds or independent biological replicates.',
            source_denominators=native['source_denominators'],native_connections=native['connections'],
            observation_scope=native['observation_scope'],upstream_bindings=native['bindings'],
            privileged_roles={'incoming':'Realized source spike delivery evidence; not admissible controls.',
                              'boundary_state':'Initial and terminal microscopic voltage; not causal model inputs.'},
            ordinary_events=int((~endpoint).sum()),final_endpoint_events=int(endpoint.sum()),
            endpoint_semantics='Raw terminal events retained; half-open training windows exclude the terminal tick.',
            admissible_visual_control_mapping_qualified=False,scientific_qualification=False,production_episodes=0)
        controls=dict(control_ticks=np.empty(0,dtype=np.int64), control_values=np.empty((0,0)), control_features=())
        if visual_controls_archive is not None:
            controls=load_visual_controls(Path(visual_controls_archive),native,native_family='pointnet')
            source_record['visual_commands']=controls.pop('source_control_record')
        episode=CircuitEpisode(case_id,circuit_id,block_id,'development',duration,dt,ticks,indices,
            controls['control_ticks'],controls['control_values'],controls['control_features'],
            seeds=seeds,source_record=source_record,privileged=private,task_outcome='not_scientifically_assessed')
        binding=dict(schema=SCHEMA,scope='source_reproduction',scenario_id='v1_visual',source_revision=revision,
            sample_unit='complete_mechanistic_episode',case_kind='constructed_pointnet_source_execution',
            native_implementation_sha256=native['implementation_sha256'],
            source_bindings=[dict(job_id=delivery['job_id'],delivery_sha256=sha256_file(native_archive/'delivery.json'))],
            coordinate_authority_delivery_sha256=sha256_file(coordinate_authority_archive/'delivery.json'),
            admissible_actions=('verified development source commands; action domain not qualified'
                if visual_controls_archive is not None else 'not qualified; no control features exported'),
            exporter_sha256=sha256_file(Path(__file__)))
        with CircuitArchiveWriter(output/'dataset',binding) as writer:
            writer.register_circuit(units);writer.add(episode)
        verification=verify_export(CircuitDataset(output/'dataset',splits=('development',)),episode,units)
        report.update(status='EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION',exit_code=0,complete_source_cases=1,
            replay_processes_verified=2,replays_counted_as_episodes=0,circuit_id=circuit_id,episode_id=case_id,
            native_job_id=delivery['job_id'],source_execution_mode=native['mode'],input_mode=native['input_mode'],
            source_units=n,at_least_100000_cortical_units=n>=100000,source_model_types=len(type_table),
            source_populations=dict(Counter(units.populations.tolist())),coordinate_units='um',
            coordinate_unit_evidence_basis=authority['unit_evidence_basis'],coordinate_preservation=coordinate_check,
            source_denominators=native['source_denominators'],observation_scope=native['observation_scope'],
            exported_declared_visual_controls=visual_controls_archive is not None,
            source_command_representation_verified=visual_controls_archive is not None,verification=verification,
            dataset_catalog_sha256=sha256_file(output/'dataset/dataset.json'))
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--native-archive',type=Path)
    source.add_argument('--pointnet-native-archive',type=Path)
    for name in ['repeatability-archive','input-replay-archive','asset-root','coordinate-authority','visual-controls-archive']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.pointnet_native_archive is not None:
        if args.coordinate_authority is None:parser.error('--coordinate-authority is required for PointNet export')
        if any(getattr(args,key) is not None for key in ('repeatability_archive','input_replay_archive','asset_root')):
            parser.error('legacy source arguments are not accepted by PointNet export')
        report=export_pointnet(args.pointnet_native_archive,args.coordinate_authority,args.output,args.visual_controls_archive)
    else:
        if args.coordinate_authority is not None:parser.error('--coordinate-authority requires --pointnet-native-archive')
        if any(getattr(args,key) is None for key in ('repeatability_archive','input_replay_archive','asset_root')):
            parser.error('--repeatability-archive, --input-replay-archive and --asset-root are required for legacy export')
        report=export(args.native_archive,args.repeatability_archive,args.input_replay_archive,args.asset_root,args.output,args.visual_controls_archive)
    print(json.dumps(report,indent=2))
    return 0 if report['status'] in ('EXPORTED_VERIFIED_V1_SOURCE_OBSERVATION','EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION') else 2


if __name__=='__main__':raise SystemExit(main())

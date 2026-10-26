"""Execute the inspected V1 graph with NEST, retaining raw synapse associations.

This reconstructs a native diagnostic from source pickles and matched GLIF3
references. It does not qualify circuit protocols, label input populations, or
count a production episode. All cortical units have a source/native identity map,
spike output (including explicit silent-unit counts), and boundary membrane state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
import zipfile

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import (ARCHIVE_BYTES, INPUT_MEMBER, NEST_VERSION, SpikeTrains,
                                  bound_json, load_pointnet_setter, prepare_inputs)
from .v1_configured_inputs import delivered_file
from .v1_lgn_spikes import load_spike_trial, replace_lgn_inputs
from .v1_nest_time_semantics import V1_DURATION_MS, V1_RESOLUTION_MS, encode_precise_times

GRAPH_MEMBER = 'GLIF_network2/network_dat.pkl'
CORTICAL_UNITS = 230924
RECURRENT_SYNAPSES = 70139111
STATE_CHECK_INTERVAL_MS = 100.0


def edge_batches(groups, source_native_ids, target_native_ids, batch_size):
    """Preserve each source/target/receptor/weight/delay tuple, without sorting."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('V1_CONNECTION_BATCH_SIZE_INVALID')
    for group in groups:
        source, target = np.asarray(group['source']), np.asarray(group['target'])
        params = group['params']
        weights, delays = np.asarray(params['weight']), np.asarray(params['delay'])
        receptor = np.asarray(params['receptor_type'])
        if (source.ndim != 1 or target.shape != source.shape or source.dtype.kind not in 'iu'
                or target.dtype.kind not in 'iu' or weights.shape != source.shape
                or weights.dtype.kind not in 'iuf' or delays.shape not in ((), source.shape)
                or delays.dtype.kind not in 'iuf' or receptor.shape != ()
                or receptor.dtype.kind not in 'iu' or not 1 <= int(receptor) <= 4):
            raise ValueError('V1_SOURCE_EDGE_SCHEMA_INVALID')
        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))
            src, dst = source[start:stop], target[start:stop]
            if (np.any(src < 0) or np.any(src >= len(source_native_ids))
                    or np.any(dst < 0) or np.any(dst >= len(target_native_ids))):
                raise ValueError('V1_SOURCE_EDGE_ENDPOINT_INVALID')
            weight = weights[start:stop].astype(np.float64, copy=True)
            delay = (np.full(stop-start, float(delays)) if delays.ndim == 0
                     else delays[start:stop].astype(np.float64, copy=True))
            if not np.isfinite(weight).all() or not np.isfinite(delay).all() or np.any(delay <= 0):
                raise ValueError('V1_SOURCE_EDGE_VALUE_INVALID')
            native_src, native_dst = source_native_ids[src], target_native_ids[dst]
            if np.any(native_src <= 0) or np.any(native_dst <= 0):
                raise ValueError('V1_SOURCE_EDGE_NODE_NOT_CREATED')
            yield {'source': native_src, 'target': native_dst, 'weight': weight,
                   'delay': delay, 'receptor_type': int(receptor)}


def connect_edges(nest, groups, source_ids, target_ids, batch_size):
    count, changed, max_difference = 0, 0, 0.0
    before = int(nest.GetKernelStatus('num_connections'))
    for batch in edge_batches(groups, source_ids, target_ids, batch_size):
        nest.Connect(batch['source'], batch['target'], conn_spec='one_to_one', syn_spec={
            'synapse_model': 'static_synapse', 'weight': batch['weight'], 'delay': batch['delay'],
            'receptor_type': batch['receptor_type']})
        effective = np.floor(batch['delay'] / V1_RESOLUTION_MS + .5) * V1_RESOLUTION_MS
        changed += int(np.count_nonzero(effective != batch['delay']))
        max_difference = max(max_difference, float(np.max(np.abs(effective-batch['delay']))))
        count += len(batch['source'])
    actual = int(nest.GetKernelStatus('num_connections')) - before
    if actual != count:
        raise ValueError('V1_NATIVE_CONNECTION_COUNT_DIFFERS')
    return {'source_synapses': count, 'native_synapses_created': actual,
            'source_delays_changed_by_released_rounding_rule': changed,
            'max_predicted_delay_change_ms': max_difference,
            'raw_tuple_order_preserved': True, 'source_weights_rescaled': False}


def native_connection_admission(nest, reference_parameters):
    """Exercise NEST's array interface before allocating the full source graph."""
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': V1_RESOLUTION_MS, 'local_num_threads': 1})
    cells = nest.Create('glif_psc', 4, params=reference_parameters)
    ids = np.asarray(cells.tolist(), dtype=np.int64)
    groups = [
        {'source': np.array([2, 0, 2, 0]), 'target': np.array([0, 2, 0, 2]),
         'params': {'receptor_type': 1, 'weight': np.array([2., -3., 0., 7.]),
                    'delay': np.array([1.7, .9, 1., 3.8])}},
        {'source': np.array([0, 3]), 'target': np.array([2, 0]),
         'params': {'receptor_type': 4, 'weight': np.array([1., -2.]), 'delay': 1.7}},
    ]
    counts = connect_edges(nest, groups, ids, ids, 3)
    actual = nest.GetConnections().get()
    receptor_key = 'receptor'
    if receptor_key not in actual:
        raise ValueError('V1_NATIVE_RECEPTOR_READBACK_MISSING:' + ','.join(actual))
    observed = sorted(zip(actual['source'], actual['target'], actual[receptor_key], actual['weight'], actual['delay']))
    expected = []
    for batch in edge_batches(groups, ids, ids, 3):
        delays = np.floor(batch['delay'] / V1_RESOLUTION_MS + .5) * V1_RESOLUTION_MS
        expected.extend(zip(batch['source'], batch['target'], [batch['receptor_type']]*len(delays), batch['weight'], delays))
    if observed != sorted(expected):
        raise ValueError('V1_NATIVE_ARRAY_CONNECTION_ASSOCIATION_DIFFERS')
    return {'status': 'PASS', **counts, 'native_tuples_exact': len(expected),
            'scope': 'Four native GLIF cells and six diagnostic connections, including parallel and zero-weight edges.'}


def load_member(archive: Path, member: str, expected_sha: str, directory: Path):
    local = directory / Path(member).name
    with zipfile.ZipFile(archive) as source:
        if source.namelist().count(member) != 1:
            raise ValueError('V1_NATIVE_SOURCE_MEMBER_NOT_UNIQUE:' + member)
        digest = hashlib.sha256()
        with source.open(member) as inp, local.open('xb') as out:
            for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                digest.update(block); out.write(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError('V1_NATIVE_SOURCE_MEMBER_CHANGED:' + member)
    with local.open('rb') as stream:
        result = pickle.load(stream)
    local.unlink()
    return result


def write_progress(output, value):
    # Live telemetry is replaceable; report.json and delivered evidence are frozen.
    (output/'progress.json').write_text(json.dumps(value, indent=2)+'\n')


def execute(source_archive, graph_inspection_archive, parameter_reference_archive,
            parameter_reconciliation_archive, input_inspection_archive, pointnet_source_archive,
            time_semantics_archive, output, threads=1, connection_batch_size=100000,
            generated_lgn_archive=None, original_input_archive=None, input_origin_archive=None):
    import nest  # Native backend is optional for pure edge-association tests.

    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'status': 'FAILED_NATIVE_SOURCE_GRAPH_EXECUTION', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'implementation_sha256': sha256_file(Path(__file__)), 'source_archive': str(source_archive),
              'source_scientific_fidelity_qualified': False, 'production_episodes': 0,
              'full_cortical_network_executed': False, 'source_population_labels_qualified': False,
              'observation_scope': 'All cortical spike events and per-unit counts; initial and terminal V_m only. No full voltage trajectory.',
              'training_runs': 0, 'threads': threads, 'connection_batch_size': connection_batch_size}
    try:
        if nest.__version__ != NEST_VERSION or type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('V1_NATIVE_RUNTIME_OR_THREAD_CONTRACT_INVALID')
        if source_archive.stat().st_size != ARCHIVE_BYTES:
            raise ValueError('V1_NATIVE_ARCHIVE_SIZE_CHANGED')
        graph, _ = bound_json(graph_inspection_archive, 'graph_identity.json')
        reference, _ = bound_json(parameter_reference_archive, 'reference/intake.json')
        reconciliation, _ = bound_json(parameter_reconciliation_archive, 'comparison.json')
        inputs, _ = bound_json(input_inspection_archive, 'input_identity.json')
        timing, _ = bound_json(time_semantics_archive, 'diagnosis/report.json')
        if (graph['cortical_neurons'] != CORTICAL_UNITS or graph['raw_synapses'] != RECURRENT_SYNAPSES
                or graph['native_metadata_identity_verified'] is not True
                or reconciliation['loader_parameters_exactly_equal'] is not True
                or reconciliation['graph_inspection_sha256'] != sha256_file(graph_inspection_archive/'graph_identity.json')
                or reconciliation['parameter_reference_sha256'] != sha256_file(parameter_reference_archive/'reference/intake.json')
                or inputs['source_config_clock'] != {'duration': V1_DURATION_MS, 'dt': V1_RESOLUTION_MS}
                or timing['source_encoding_comparison']['all_event_pairs_exact'] is not True
                or timing['native_witness']['status'] != 'PASS'):
            raise ValueError('V1_NATIVE_PREREQUISITE_EVIDENCE_DIFFERS')
        by_type = {row['specimen_id']: row for row in reference['references']}
        if len(by_type) != 111 or len(graph['node_groups']) != 111:
            raise ValueError('V1_NATIVE_CELL_TYPE_COHORT_DIFFERS')
        report['native_connection_admission'] = native_connection_admission(nest, next(iter(by_type.values()))['converted_parameters'])
        write_progress(output, dict(report, status='NATIVE_ARRAY_CONNECTION_ADMISSION_PASS'))
        with tempfile.TemporaryDirectory(prefix='v1-source-network-') as temp:
            raw = load_member(source_archive, GRAPH_MEMBER, graph['member_sha256'][GRAPH_MEMBER], Path(temp))
            raw_inputs = load_member(source_archive, INPUT_MEMBER, inputs['input_member_sha256'], Path(temp))
        if generated_lgn_archive is not None:
            if original_input_archive is None or input_origin_archive is None:
                raise ValueError('V1_GENERATED_LGN_REQUIRES_ORIGINAL_INPUT_ROLE_BINDING')
            generated, spike_report = load_spike_trial(generated_lgn_archive)
            if (spike_report['source_units'] != 17400
                    or spike_report['config']['unit_coverage'] != 'all_source_units'):
                raise ValueError('V1_GENERATED_LGN_REQUIRES_COMPLETE_SOURCE_POPULATION')
            origin, _ = bound_json(input_origin_archive, 'comparison.json')
            original_path = delivered_file(original_input_archive, 'native_replay/source_inputs.npz')
            if sha256_file(original_path) != origin['source_plan_sha256']:
                raise ValueError('V1_LGN_INPUT_ORIGIN_PLAN_CHANGED')
            with np.load(original_path, allow_pickle=False) as source:
                original_plan = {key: source[key] for key in source.files}
            raw_inputs, replacement = replace_lgn_inputs(raw_inputs, generated, original_plan, origin)
            report['generated_lgn'] = dict(replacement, source_archive=str(generated_lgn_archive),
                spike_report_sha256=sha256_file(generated_lgn_archive/'spikes/report.json'),
                input_identity_sha256=spike_report['trials'][0]['input_identity_sha256'], trial=0,
                original_plan_sha256=sha256_file(original_path),
                input_origin_sha256=sha256_file(input_origin_archive/'comparison.json'),
                config=spike_report['config'],
                interpretation='Prospective source LGN input with the fixed legacy background; not historical recipe reproduction.')
        elif original_input_archive is not None or input_origin_archive is not None:
            raise ValueError('V1_INPUT_OVERRIDE_BINDINGS_REQUIRE_GENERATED_LGN')
        report['input_loading_seconds'] = time.perf_counter() - start
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': V1_RESOLUTION_MS, 'local_num_threads': threads})
        report.update(nest_version=nest.__version__, kernel_rng_seed=int(nest.GetKernelStatus('rng_seed')))
        native_ids = np.zeros(CORTICAL_UNITS, dtype=np.int64)
        type_ids = np.zeros(CORTICAL_UNITS, dtype=np.int64)
        expected_vm = np.empty(CORTICAL_UNITS, dtype=np.float64)
        if len(raw['nodes']) != len(graph['node_groups']):
            raise ValueError('V1_NATIVE_NODE_GROUP_COUNT_DIFFERS')
        for index, (group, identity) in enumerate(zip(raw['nodes'], graph['node_groups'])):
            ids = np.asarray(group['ids'])
            if (identity['raw_group_index'] != index or len(ids) != identity['neurons']
                    or ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
                    or np.any(ids >= CORTICAL_UNITS) or np.any(native_ids[ids])
                    or len(np.unique(ids)) != len(ids)):
                raise ValueError('V1_NATIVE_SOURCE_NODE_GROUP_DIFFERS')
            specimen = identity['native_node_type_id']
            params = by_type[specimen]['converted_parameters']
            if group['params']['V_dynamics_method'] != 'linear_exact' or not np.array_equal(group['params']['asc_init'], params['asc_init']):
                raise ValueError('V1_NATIVE_SOURCE_INITIALIZATION_DIFFERS')
            cells = nest.Create('glif_psc', len(ids), params=params)
            native_ids[ids] = cells.tolist(); type_ids[ids] = specimen; expected_vm[ids] = params['V_m']
        if np.any(native_ids <= 0) or len(np.unique(native_ids)) != CORTICAL_UNITS:
            raise ValueError('V1_NATIVE_CORTICAL_MAPPING_INCOMPLETE')
        sort_order = np.argsort(native_ids)
        ordered_ids = native_ids[sort_order]
        cortical = nest.NodeCollection(ordered_ids.tolist())
        initial_vm = np.asarray(cortical.get('V_m'), dtype=np.float64)
        if not np.array_equal(initial_vm, expected_vm[sort_order]):
            raise ValueError('V1_NATIVE_CORTICAL_INITIAL_VM_DIFFERS')
        np.savez_compressed(output/'cortical_identity.npz', source_id=np.arange(CORTICAL_UNITS),
                            native_id=native_ids, node_type_id=type_ids, initial_vm_mV=expected_vm)
        report['recurrent_connections'] = connect_edges(nest, raw['edges'], native_ids, native_ids, connection_batch_size)
        if report['recurrent_connections']['source_synapses'] != RECURRENT_SYNAPSES:
            raise ValueError('V1_NATIVE_RECURRENT_DENOMINATOR_DIFFERS')
        del raw
        write_progress(output, dict(report, status='NATIVE_RECURRENT_GRAPH_CREATED'))
        plan = prepare_inputs(raw_inputs, V1_DURATION_MS)
        if generated_lgn_archive is not None:
            np.savez_compressed(output/'source_inputs.npz', **{key: value for key, value in plan.items() if key != 'trains'})
        generators = nest.Create('spike_generator', len(plan['trains']), params={'precise_times': True})
        input_native_ids = np.asarray(generators.tolist(), dtype=np.int64)
        setter = load_pointnet_setter(pointnet_source_archive, nest)
        raw_time_mismatches = 0
        for index, train in enumerate(plan['trains']):
            node = nest.NodeCollection([int(input_native_ids[index])])
            setter(int(plan['source_id'][index]), node, SpikeTrains(train), t_offset=0.)
            observed = np.asarray(nest.GetStatus(node, 'spike_times')[0])
            sorted_train = np.sort(train)
            encoded = encode_precise_times(sorted_train, float(nest.GetKernelStatus('tics_per_ms')))
            if not np.array_equal(observed, encoded['decoded_times_ms']):
                raise ValueError('V1_NATIVE_INPUT_SOURCE_ENCODING_DIFFERS')
            raw_time_mismatches += int(np.count_nonzero(observed != sorted_train))
        report['input_connections'] = []
        for index, (nodes, edges) in enumerate(raw_inputs):
            selected = plan['population'] == index
            mapping = np.zeros(int(selected.sum()), dtype=np.int64)
            mapping[plan['source_id'][selected]] = input_native_ids[selected]
            result = connect_edges(nest, edges, mapping, native_ids, connection_batch_size)
            if (result['source_synapses'] != inputs['populations'][index]['raw_edges']
                    or len(mapping) != inputs['populations'][index]['input_units']):
                raise ValueError('V1_NATIVE_INPUT_CONNECTION_DENOMINATOR_DIFFERS')
            report['input_connections'].append(dict(result, population_index=index))
        report.update(input_generators=len(input_native_ids), input_events=int(plan['source_counts'].sum()),
                      source_input_timestamp_roundtrips_differ=raw_time_mismatches)
        np.savez_compressed(output/'input_identity.npz', population=plan['population'], source_id=plan['source_id'],
                            native_id=input_native_ids, source_event_counts=plan['source_counts'])
        del raw_inputs, plan
        recorder = nest.Create('spike_recorder')
        nest.Connect(cortical, recorder, syn_spec={'delay': V1_RESOLUTION_MS})
        report['setup_elapsed_seconds'] = time.perf_counter() - start
        report['native_total_connections'] = int(nest.GetKernelStatus('num_connections'))
        report['kernel_delays_ms'] = dict(zip(['min_delay','max_delay'], nest.GetKernelStatus(['min_delay','max_delay'])))
        report['state_check_times_ms'] = []
        before = time.perf_counter()
        for stop in np.arange(STATE_CHECK_INTERVAL_MS, V1_DURATION_MS+STATE_CHECK_INTERVAL_MS, STATE_CHECK_INTERVAL_MS):
            nest.Simulate(STATE_CHECK_INTERVAL_MS)
            state = np.asarray(cortical.get('V_m'), dtype=np.float64)
            if state.shape != (CORTICAL_UNITS,) or not np.isfinite(state).all():
                raise ValueError('V1_NATIVE_NONFINITE_CORTICAL_STATE')
            report['state_check_times_ms'].append(float(stop))
            write_progress(output, dict(report, status='NATIVE_SOURCE_GRAPH_RUNNING', simulated_ms=float(stop)))
        report['simulation_seconds'] = time.perf_counter() - before
        events = recorder.get('events')
        senders, times = np.asarray(events['senders']), np.asarray(events['times'], dtype=np.float64)
        positions = np.searchsorted(ordered_ids, senders)
        if (np.any(positions >= CORTICAL_UNITS) or not np.array_equal(ordered_ids[positions], senders)
                or not np.isfinite(times).all() or np.any(times <= 0) or np.any(times > V1_DURATION_MS)):
            raise ValueError('V1_NATIVE_CORTICAL_SPIKE_OUTPUT_INVALID')
        source_ids = sort_order[positions]
        counts = np.bincount(source_ids, minlength=CORTICAL_UNITS)
        final_vm = np.empty(CORTICAL_UNITS, dtype=np.float64); final_vm[sort_order] = state
        np.savez_compressed(output/'cortical_activity.npz', spike_source_id=source_ids, spike_time_ms=times,
                            spike_counts=counts, final_vm_mV=final_vm)
        report.update(status='NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE', full_cortical_network_executed=True,
                      coupled_cortical_neurons=CORTICAL_UNITS, source_topology_reused=graph['topology'],
                      native_duration_ms=float(nest.GetKernelStatus('biological_time')), cortical_spikes=len(times),
                      silent_cortical_units=int(np.count_nonzero(counts==0)),
                      mean_firing_rate_hz=float(counts.mean()/(V1_DURATION_MS/1000.)),
                      max_firing_rate_hz=float(counts.max()/(V1_DURATION_MS/1000.)),
                      time_semantics_evidence=str(time_semantics_archive),
                      parameter_initialization_source='Matched Allen GLIF3 converted references; raw nine-parameter equality and zero ASC initialization retained.')
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        report['artifacts'] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'progress.json'}
        freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-archive','graph-inspection-archive','parameter-reference-archive','parameter-reconciliation-archive',
                 'input-inspection-archive','pointnet-source-archive','time-semantics-archive','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('generated-lgn-archive', 'original-input-archive', 'input-origin-archive'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--connection-batch-size', type=int, default=100000)
    report = execute(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2))
    return 0 if report['status']=='NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Replay all archived V1 input generators through the pinned PointNet/NEST API.

This isolates source input delivery. No cortical network is simulated, no source
population labels are inferred, and these runs never count as production episodes.
Timestamp equality is measured exactly; no fitted numerical tolerance is applied.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from .provenance import freeze_json, sha256_file

NEST_VERSION = '3.10.0'
POINTNET_REVISION = '564fe78e24cb7d57f439f915863893093ce6e16b'
POINTNETWORK_SHA256 = '92592d5b2af5d27aed7847f920d45033858e67e2cf7f8d095316d38d77656bcc'
POINTSIMULATOR_SHA256 = '3b104b5fdf83824c381b186667af31f55823c8b008471ab5d8fa0785f884827a'
ARCHIVE_BYTES = 3423419377
INPUT_MEMBER = 'GLIF_network2/input_dat.pkl'


def bound_json(archive, relative):
    archive = Path(archive)
    delivery = json.loads((archive/'delivery.json').read_text())
    rows = [row for row in delivery['files'] if row['path'] == relative]
    path = archive/relative
    if len(rows) != 1 or path.stat().st_size != rows[0]['bytes'] or sha256_file(path) != rows[0]['sha256']:
        raise ValueError('V1_INPUT_REPLAY_RECEIPT_CHANGED:'+relative)
    return json.loads(path.read_text()), delivery


def prepare_inputs(raw, duration):
    """Retain full source identities, silent generators and repeated timestamps."""
    if not isinstance(raw, (list, tuple)) or not raw or not np.isfinite(duration) or duration <= 0:
        raise ValueError('V1_INPUT_REPLAY_POPULATIONS_OR_DURATION_INVALID')
    populations, source_ids, trains = [], [], []
    for population, entry in enumerate(raw):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError('V1_INPUT_REPLAY_POPULATION_SCHEMA_INVALID')
        nodes, _ = entry
        ids = np.asarray(nodes['ids'])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                or not np.array_equal(np.sort(ids), np.arange(len(ids)))
                or len(nodes['spikes']) != len(ids)):
            raise ValueError('V1_INPUT_REPLAY_SOURCE_IDENTITIES_INVALID')
        for source_id, values in zip(ids, nodes['spikes']):
            values = np.asarray(values)
            if (values.ndim != 1 or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()
                    or np.any(values <= 0) or np.any(values > duration)):
                raise ValueError('V1_INPUT_REPLAY_SPIKE_TIME_INVALID')
            populations.append(population)
            source_ids.append(int(source_id))
            trains.append(values.astype(np.float64, copy=True))
    counts = np.asarray([len(train) for train in trains], dtype=np.int64)
    return dict(population=np.asarray(populations,dtype=np.int64),
        source_id=np.asarray(source_ids,dtype=np.int64), trains=trains,
        source_event_unit=np.repeat(np.arange(len(trains),dtype=np.int64), counts),
        source_event_time_ms=np.concatenate(trains), source_counts=counts)


def compare_events(plan, native_ids, events):
    """Match sorted event pairs, preserving multiplicity and the population map."""
    native_ids = np.asarray(native_ids)
    senders, times = np.asarray(events['senders']), np.asarray(events['times'])
    units = len(plan['source_id'])
    if (native_ids.shape != (units,) or native_ids.dtype.kind not in 'iu' or np.any(native_ids<=0)
            or len(np.unique(native_ids)) != units
            or senders.ndim != 1 or senders.dtype.kind not in 'iu'
            or times.shape != senders.shape or times.dtype.kind not in 'iuf'
            or not np.isfinite(times).all()):
        raise ValueError('V1_NATIVE_INPUT_EVENT_SCHEMA_OR_MAPPING_INVALID')
    order = np.argsort(native_ids)
    found = np.searchsorted(native_ids[order], senders)
    if np.any(found >= units) or not np.array_equal(native_ids[order][found], senders):
        raise ValueError('V1_NATIVE_INPUT_UNKNOWN_SENDER')
    event_units = order[found]
    counts = np.bincount(event_units, minlength=units)
    count_equal = np.array_equal(counts, plan['source_counts'])
    mismatched = np.flatnonzero(counts != plan['source_counts'])
    result = dict(mapped_generators=units,silent_source_generators=int(np.count_nonzero(plan['source_counts']==0)),
        expected_events=len(plan['source_event_time_ms']),observed_events=len(times),
        every_generator_count_exact=bool(count_equal),mismatched_generator_counts=len(mismatched),
        count_mismatch_examples=[dict(population=int(plan['population'][i]),source_id=int(plan['source_id'][i]),
            expected=int(plan['source_counts'][i]),observed=int(counts[i])) for i in mismatched[:8]],
        timestamps_exact=False, timestamp_mismatches=None, max_absolute_time_difference_ms=None,
        equality_rule='Exact sorted source/native timestamp equality per generator; no fitted tolerance.')
    if count_equal:
        expected_order = np.lexsort((plan['source_event_time_ms'],plan['source_event_unit']))
        actual_order = np.lexsort((times,event_units))
        if not np.array_equal(plan['source_event_unit'][expected_order],event_units[actual_order]):
            raise ValueError('V1_NATIVE_INPUT_PAIRING_INVALID')
        expected = plan['source_event_time_ms'][expected_order]
        actual = times[actual_order]
        different = np.flatnonzero(expected != actual)
        result.update(timestamps_exact=not len(different),timestamp_mismatches=len(different),
            max_absolute_time_difference_ms=float(np.max(np.abs(actual-expected))) if len(expected) else 0.,
            timestamp_mismatch_examples=[dict(population=int(plan['population'][plan['source_event_unit'][expected_order[i]]]),
                source_id=int(plan['source_id'][plan['source_event_unit'][expected_order[i]]]),
                source_time_ms=float(expected[i]),native_time_ms=float(actual[i])) for i in different[:8]])
    result['all_event_pairs_exact'] = bool(count_equal and result['timestamps_exact'])
    return result


def load_pointnet_setter(source_archive, nest):
    source_archive = Path(source_archive)
    info, delivery = bound_json(source_archive,'source_inspection.json')
    if (delivery['status'] != 'PINNED_POINTNET_SOURCE_ACQUIRED'
            or info['bmtk_revision'] != POINTNET_REVISION
            or info['converter_matches_acquired_111_references'] is not True):
        raise ValueError('V1_PINNED_POINTNET_SOURCE_REQUIRED')
    network = source_archive/'source/bmtk/simulator/pointnet/pointnetwork.py'
    simulator = source_archive/'source/bmtk/simulator/pointnet/pointsimulator.py'
    if sha256_file(network) != POINTNETWORK_SHA256 or sha256_file(simulator) != POINTSIMULATOR_SHA256:
        raise ValueError('V1_POINTNET_INPUT_IMPLEMENTATION_CHANGED')
    tree = ast.parse(network.read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='set_spikes_nest3']
    if len(functions) != 1:
        raise ValueError('V1_POINTNET_SET_SPIKES_FUNCTION_MISSING')
    def fail(message):
        raise ValueError(message)
    namespace = dict(np=np,nest=nest,io=SimpleNamespace(log_exception=fail))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(network),'exec'),namespace)
    return namespace['set_spikes_nest3']


class SpikeTrains:
    def __init__(self, train):
        self.train = train

    def get_times(self, node_id):
        return self.train.copy()


def execute_input_plan(nest, plan, setter, output, report, dt, duration, started):
    """Execute a complete generator plan through the shared native replay path."""
    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=dt,local_num_threads=1))
    report['kernel_rng_seed'] = int(nest.GetKernelStatus('rng_seed'))
    generators = nest.Create('spike_generator',len(plan['trains']),params={'precise_times':True})
    native_ids = np.asarray(generators.tolist(),dtype=np.int64)
    np.savez_compressed(output/'source_inputs.npz',population=plan['population'],source_id=plan['source_id'],
        native_id=native_ids,source_event_unit=plan['source_event_unit'],
        source_event_time_ms=plan['source_event_time_ms'],source_counts=plan['source_counts'])
    loaded_exact = True
    for index,train in enumerate(plan['trains']):
        node = nest.NodeCollection([int(native_ids[index])])
        setter(int(plan['source_id'][index]),node,SpikeTrains(train),t_offset=0.)
        stored = np.asarray(nest.GetStatus(node,'spike_times')[0])
        loaded_exact = loaded_exact and np.array_equal(stored,np.sort(train))
    report['requested_spike_times_read_back_exact'] = bool(loaded_exact)
    recorder = nest.Create('spike_recorder')
    nest.Connect(generators,recorder,syn_spec={'delay':dt})
    report['setup_elapsed_seconds'] = time.perf_counter()-started
    before = time.perf_counter();nest.Simulate(duration)
    report['source_duration_simulation_seconds'] = time.perf_counter()-before
    report['native_input_simulation_executed'] = True
    events = {k:np.asarray(v).copy() for k,v in nest.GetStatus(recorder,'events')[0].items()}
    np.savez_compressed(output/'events_at_source_stop.npz',**events)
    report['at_source_stop'] = compare_events(plan,native_ids,events)
    # A separately declared recorder-only continuation diagnoses delayed
    # collection. It never redefines the original source duration.
    nest.Simulate(dt)
    flushed = {k:np.asarray(v).copy() for k,v in nest.GetStatus(recorder,'events')[0].items()}
    np.savez_compressed(output/'events_after_recorder_flush.npz',**flushed)
    report['after_recorder_flush'] = compare_events(plan,native_ids,flushed)
    report['native_total_duration_ms'] = float(nest.GetKernelStatus('biological_time'))
    report['native_kernel_delays_ms'] = dict(zip(['min_delay','max_delay'],nest.GetKernelStatus(['min_delay','max_delay'])))
    report['status'] = ('NATIVE_INPUT_EVENT_REPLAY_EXACT' if loaded_exact and report['at_source_stop']['all_event_pairs_exact']
                        else 'NATIVE_INPUT_EVENTS_REQUIRE_RECONCILIATION')
    return report


def replay(source_archive, input_inspection_archive, pointnet_source_archive, output):
    # NEST is an optional native runtime; pure comparator tests run without it.
    import nest

    source_archive, input_inspection_archive, pointnet_source_archive, output = map(
        Path,(source_archive,input_inspection_archive,pointnet_source_archive,output))
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='FAILED_NATIVE_INPUT_EXECUTION',started_utc=datetime.now(timezone.utc).isoformat(),
        source_archive=str(source_archive),input_inspection_archive=str(input_inspection_archive),
        pointnet_source_archive=str(pointnet_source_archive),nest_version=nest.__version__,
        implementation_sha256=sha256_file(Path(__file__)),native_input_simulation_executed=False,
        full_cortical_network_executed=False,source_population_labels_independently_qualified=False,
        source_scientific_fidelity_qualified=False,production_episodes=0)
    started = time.perf_counter()
    try:
        if nest.__version__ != NEST_VERSION:
            raise ValueError('V1_INPUT_REPLAY_NEST_VERSION_DIFFERS')
        inspection, delivered = bound_json(input_inspection_archive,'input_identity.json')
        if (delivered['status'] != 'INSPECTED' or delivered['exit_code'] != 0
                or inspection['status'] != 'SOURCE_INPUT_INSPECTED'
                or inspection['input_member'] != INPUT_MEMBER
                or source_archive.stat().st_size != ARCHIVE_BYTES):
            raise ValueError('V1_COMPLETE_INSPECTED_SOURCE_INPUT_REQUIRED')
        clock = inspection['source_config_clock']
        if clock != {'duration':3000.0,'dt':0.25}:
            raise ValueError('V1_ORIGINAL_POINTNET_CLOCK_CHANGED')
        dt, duration = clock['dt'], clock['duration']
        with tempfile.TemporaryDirectory(prefix='v1-native-inputs-') as temp:
            local = Path(temp)/'input_dat.pkl'
            with zipfile.ZipFile(source_archive) as archive:
                if archive.namelist().count(INPUT_MEMBER) != 1:
                    raise ValueError('V1_INPUT_ARCHIVE_MEMBER_NOT_UNIQUE')
                digest = hashlib.sha256()
                with archive.open(INPUT_MEMBER) as src, local.open('xb') as dst:
                    for block in iter(lambda:src.read(8*1024*1024),b''):
                        digest.update(block);dst.write(block)
            if digest.hexdigest() != inspection['input_member_sha256']:
                raise ValueError('V1_INSPECTED_INPUT_MEMBER_CHANGED')
            with local.open('rb') as stream:raw=pickle.load(stream)
        plan = prepare_inputs(raw,duration)
        del raw
        for index,population in enumerate(inspection['populations']):
            selected = plan['population']==index
            if (int(selected.sum()) != population['input_units']
                    or int(plan['source_counts'][selected].sum()) != population['spikes']['raw_events']):
                raise ValueError('V1_INPUT_REPLAY_INSPECTED_DENOMINATOR_DIFFERS')
        if len(np.unique(plan['population'])) != len(inspection['populations']):
            raise ValueError('V1_INPUT_REPLAY_POPULATION_COUNT_DIFFERS')
        setter = load_pointnet_setter(pointnet_source_archive,nest)
        report.update(source_config_clock=clock,pointnet_revision=POINTNET_REVISION,
            pointnet_spike_generator_parameters={'precise_times':True},source_time_offset_ms=0.,
            pointnet_setter='set_spikes_nest3',pointnetwork_sha256=POINTNETWORK_SHA256,
            pointsimulator_sha256=POINTSIMULATOR_SHA256,
            source_input_member_sha256=digest.hexdigest(),
            input_inspection_sha256=sha256_file(input_inspection_archive/'input_identity.json'),
            observation_scope='Input generators connected only to a diagnostic recorder; no cortical or source synapses.',
            diagnostic_recorder_delay_ms=dt,predeclared_recorder_flush_ms=dt)
        execute_input_plan(nest, plan, setter, output, report, dt, duration, started)
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        report['artifacts']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
        freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-archive','input-inspection-archive','pointnet-source-archive','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    result=replay(**vars(parser.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts']},indent=2))
    return 0 if result['status']=='NATIVE_INPUT_EVENT_REPLAY_EXACT' else 2


if __name__=='__main__':
    raise SystemExit(main())

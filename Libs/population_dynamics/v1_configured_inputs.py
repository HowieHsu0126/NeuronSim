"""Compare configured V1 source inputs with archived raw generator events.

This reads existing evidence; it never resamples inputs or runs the network.
The configured stimulus stream is distinct from a visual action interface.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import bound_json, compare_events


def validate_plan(plan):
    """Check the complete generator inventory and its event/count agreement."""
    population, source_id, counts, event_unit = [np.asarray(plan[key]) for key in
        ('population', 'source_id', 'source_counts', 'source_event_unit')]
    times = np.asarray(plan['source_event_time_ms'])
    n = len(source_id)
    if (not n or any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in
                     (population, source_id, counts, event_unit))
            or population.shape != (n,) or counts.shape != (n,)
            or np.any(population < 0) or np.any(source_id < 0) or np.any(counts < 0)
            or times.shape != event_unit.shape or times.dtype.kind not in 'iuf'
            or not np.isfinite(times).all() or np.any(times <= 0)
            or np.any(event_unit < 0) or np.any(event_unit >= n)):
        raise ValueError('V1_CONFIGURED_INPUT_PLAN_INVALID')
    pairs = np.rec.fromarrays([population, source_id])
    if len(np.unique(pairs)) != n:
        raise ValueError('V1_CONFIGURED_INPUT_PLAN_DUPLICATE_IDENTITY')
    if not np.array_equal(np.bincount(event_unit.astype(np.int64), minlength=n), counts):
        raise ValueError('V1_CONFIGURED_INPUT_PLAN_COUNTS_DIFFER')


def compare_population(plan, population, csv_ids, csv_times):
    """Compare exact event multisets for an explicit candidate population join."""
    validate_plan(plan)
    selected = np.flatnonzero(plan['population'] == population)
    if not len(selected):
        raise ValueError('V1_CONFIGURED_INPUT_POPULATION_ABSENT')
    ids, times = np.asarray(csv_ids), np.asarray(csv_times)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or times.shape != ids.shape or times.dtype.kind not in 'iuf'
            or not np.isfinite(times).all() or np.any(times <= 0)):
        raise ValueError('V1_CONFIGURED_INPUT_CSV_EVENTS_INVALID')
    source_ids = plan['source_id'][selected]
    source_order = np.argsort(source_ids)
    position = np.searchsorted(source_ids[source_order], ids)
    bounded = np.minimum(position, len(source_ids) - 1)
    known = (position < len(source_ids)) & (source_ids[source_order][bounded] == ids)
    unknown_ids, unknown_counts = np.unique(ids[~known], return_counts=True)
    local_unit = np.full(len(plan['source_id']), -1, dtype=np.int64)
    local_unit[selected] = np.arange(len(selected))
    source_events = plan['population'][plan['source_event_unit']] == population
    local_plan = dict(population=plan['population'][selected], source_id=source_ids,
                      source_counts=plan['source_counts'][selected],
                      source_event_unit=local_unit[plan['source_event_unit'][source_events]],
                      source_event_time_ms=plan['source_event_time_ms'][source_events])
    # Reuse the already-tested exact comparator. These positive integer IDs are
    # a local join key only; they do not claim to be NEST or source node IDs.
    join_ids = np.arange(1, len(selected) + 1, dtype=np.int64)
    events = dict(senders=source_order[position[known]] + 1, times=times[known])
    result = compare_events(local_plan, join_ids, events)
    result.update(candidate_archive_population=int(population),
                  csv_events=int(len(ids)), csv_active_source_ids=int(len(np.unique(ids))),
                  unknown_csv_events=int((~known).sum()),
                  unknown_csv_source_ids=[dict(source_id=int(i), events=int(c))
                                          for i, c in zip(unknown_ids, unknown_counts)])
    result['all_event_pairs_exact'] = bool(result['all_event_pairs_exact'] and known.all())
    return result


def delivered_file(archive, relative):
    archive = Path(archive)
    delivery = json.loads((archive / 'delivery.json').read_text())
    rows = [row for row in delivery['files'] if row['path'] == relative]
    path = archive / relative
    if (len(rows) != 1 or path.stat().st_size != rows[0]['bytes']
            or sha256_file(path) != rows[0]['sha256']):
        raise ValueError('V1_CONFIGURED_INPUT_ARCHIVE_CHANGED:' + relative)
    return path


def read_csv_events(path, duration_ms):
    # This is the header-present path in the pinned PointNet CSVSTReader.
    # Keep its pandas parser and file-scoped IDs; do not relabel population.
    frame = pd.read_csv(path, sep=' ', header=0)
    if list(frame.columns) != ['timestamps', 'population', 'node_ids'] or not len(frame):
        raise ValueError('V1_CONFIGURED_CSV_HEADER_OR_EMPTY_INPUT')
    ids, times, labels = [frame[key].to_numpy(copy=True) for key in
                          ('node_ids', 'timestamps', 'population')]
    if (ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or times.dtype.kind not in 'iuf' or not np.isfinite(times).all()
            or np.any(times <= 0) or np.any(times > duration_ms)
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError('V1_CONFIGURED_CSV_VALUES_INVALID')
    if len(set(labels)) != 1:
        raise ValueError('V1_CONFIGURED_CSV_MULTIPLE_SOURCE_POPULATIONS')
    return ids, times, labels



def read_h5_events(path, duration_ms):
    """Read the two SONATA layouts supported by the pinned BMTK reader.

    Missing units use that reader's explicit ms default. A declared different
    unit is rejected, never silently rescaled. Unnamed legacy populations stay
    unnamed; the target population comes separately from the source config.
    """
    with h5py.File(path, 'r') as source:
        root = source['spikes']
        groups = [key for key, value in root.items() if isinstance(value, h5py.Group)]
        if groups:
            if len(groups) != 1:
                raise ValueError('V1_CONFIGURED_H5_MULTIPLE_SOURCE_POPULATIONS')
            group, id_name, labels = root[groups[0]], 'node_ids', groups
            layout = 'population_group'
        else:
            group, id_name, labels = root, 'gids', []
            layout = 'legacy_unnamed_population'
        if id_name not in group or 'timestamps' not in group:
            raise ValueError('V1_CONFIGURED_H5_EVENT_DATASETS_MISSING')
        ids, times = group[id_name][:], group['timestamps'][:]
        unit = group['timestamps'].attrs.get('units')
        if isinstance(unit, bytes): unit = unit.decode('utf-8')
        if unit is not None and unit != 'ms':
            raise ValueError('V1_CONFIGURED_H5_TIME_UNIT_DIFFERS')
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or times.shape != ids.shape or times.dtype.kind not in 'iuf'
            or not np.isfinite(times).all() or np.any(times <= 0) or np.any(times > duration_ms)):
        raise ValueError('V1_CONFIGURED_H5_VALUES_INVALID')
    return ids, times, dict(source_population_labels=labels, source_layout=layout,
                            declared_time_units=unit, interpreted_time_units='ms',
                            unit_basis='timestamps attribute' if unit is not None else 'pinned BMTK reader default')

def read_input_inventory(nodes_path, types_path, node_set):
    with types_path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter=' ', skipinitialspace=True))
    types = {int(row['node_type_id']): row for row in rows}
    if len(types) != len(rows) or not types:
        raise ValueError('V1_INPUT_NODE_TYPES_INVALID')
    with h5py.File(nodes_path, 'r') as source:
        if set(source['nodes']) != {node_set}:
            raise ValueError('V1_INPUT_NODE_SET_DIFFERS')
        group = source['nodes'][node_set]
        ids, type_ids = group['node_id'][:], group['node_type_id'][:]
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or type_ids.shape != ids.shape or type_ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids) or not set(map(int, type_ids)).issubset(types)):
        raise ValueError('V1_INPUT_NODE_IDENTITIES_INVALID')
    return ids, dict(node_set=node_set, source_nodes=len(ids),
                     declared_node_types=len(types), observed_node_types=int(len(np.unique(type_ids))),
                     type_records=rows)


def compare_archive(csv_archive, raw_archive, output):
    report = dict(status='FAILED_SOURCE_INPUT_COMPARISON', source_fidelity_qualified=False,
                  original_config_equivalence_qualified=False, production_episodes=0,
                  new_simulations=0, started_utc=datetime.now(timezone.utc).isoformat(),
                  implementation_sha256=sha256_file(Path(__file__)), cases=[])
    try:
        intake, delivery = bound_json(csv_archive, 'intake.json')
        raw_report, _ = bound_json(raw_archive, 'native_replay/report.json')
        raw_path = delivered_file(raw_archive, 'native_replay/source_inputs.npz')
        with np.load(raw_path, allow_pickle=False) as source:
            plan = {key: source[key] for key in source.files}
        validate_plan(plan)
        clock = raw_report['source_config_clock']
        if clock != {'duration': 3000., 'dt': .25}:
            raise ValueError('V1_CONFIGURED_SOURCE_CLOCK_DIFFERS')
        if set(np.unique(plan['population']).tolist()) != {0, 1}:
            raise ValueError('V1_CONFIGURED_SOURCE_POPULATIONS_DIFFER')
        config_path = delivered_file(csv_archive, 'point_config.json')
        config = json.loads(config_path.read_text())
        if {key: config['run'][key] for key in ('duration', 'dt')} != clock:
            raise ValueError('V1_CONFIGURED_SOURCE_CONFIG_CLOCK_DIFFERS')
        if len(intake['cases']) != 2 or {row['input_key'] for row in intake['cases']} != set(config['inputs']):
            raise ValueError('V1_CONFIGURED_SOURCE_CASE_DENOMINATOR_DIFFERS')
        reader = delivered_file(csv_archive, 'reader/spike_train_readers.py')
        if sha256_file(reader) != intake['csv_reader_sha256'] or intake['csv_reader_sha256'] != '1bb1f542b54e95426ff1ba8ea46827649c1c1caee8f8122ff92cc785422a25ac':
            raise ValueError('V1_CONFIGURED_CSV_READER_SOURCE_CHANGED')
        report.update(csv_intake_sha256=sha256_file(csv_archive/'intake.json'),
                      source_plan_sha256=sha256_file(raw_path),
                      source_csv_reader_sha256=intake['csv_reader_sha256'],
                      parser='Pinned BMTK header-present CSV and single-population SONATA H5 layouts; no resampling or time-unit conversion',
                      pandas_version=pd.__version__, source_clock=clock, source_run_configuration=config['run'],
                      original_lgn_condition_acquired=intake['original_lgn_condition_acquired'])
        for case in intake['cases']:
            key = case['input_key']; source_config = config['inputs'][key]
            population = case['candidate_archive_population']
            source_name = case.get('source_events', case.get('source_csv'))
            source_path = delivered_file(csv_archive, source_name)
            source_format = source_config['module']
            if source_format == 'csv':
                ids, times, labels = read_csv_events(source_path, clock['duration'])
                event_metadata = dict(source_population_labels=sorted(set(labels)),
                                      interpreted_time_units='ms', unit_basis='pinned BMTK CSV reader default')
            elif source_format == 'h5':
                ids, times, event_metadata = read_h5_events(source_path, clock['duration'])
                labels = event_metadata['source_population_labels']
            else:
                raise ValueError('V1_CONFIGURED_SOURCE_FORMAT_UNSUPPORTED')
            result = compare_population(plan, population, ids, times)
            result.update(input_key=key, source_config=source_config,
                          candidate_role=case['candidate_role'], source_events=source_name,
                          source_events_sha256=sha256_file(source_path), source_format=source_format,
                          event_metadata=event_metadata,
                          node_inventory_role=case.get('node_inventory_role', 'Current public PointNet metadata'),
                          target_node_set=source_config['node_set'],
                          source_population_label_equals_target=bool(set(labels) == {source_config['node_set']}),
                          pointnet_join='Source add_spike_trains chooses target nodes by node_set; '
                            'set_spikes_nest3 calls get_times(node_id=...), with no population filter. '
                            'CSV labels are preserved; file-scoped node IDs are compared.',
                          input_generator_inventory_exact=False)
            nodes = delivered_file(csv_archive, case['nodes_file'])
            types = delivered_file(csv_archive, case['types_file'])
            inventory, description = read_input_inventory(nodes, types, source_config['node_set'])
            archived = plan['source_id'][plan['population'] == population]
            result.update(input_inventory=description,
                          input_generator_inventory_exact=bool(np.array_equal(np.sort(archived), np.sort(inventory))),
                          nodes_sha256=sha256_file(nodes), types_sha256=sha256_file(types),
                          csv_ids_within_source_inventory=bool(np.isin(ids, inventory).all()))
            report['cases'].append(result)
        report['candidate_event_and_inventory_identity_exact'] = all(
            row['all_event_pairs_exact'] and row['input_generator_inventory_exact']
            and row['csv_ids_within_source_inventory'] for row in report['cases'])
        report['status'] = ('SOURCE_INPUT_CANDIDATES_EXACT' if report['candidate_event_and_inventory_identity_exact']
                            else 'SOURCE_INPUT_CANDIDATES_DIFFER')
        report['visual_action_mapping_qualified'] = False
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('csv-archive', 'raw-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_archive(args.csv_archive, args.raw_archive, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))
    if result['status'] != 'SOURCE_INPUT_CANDIDATES_EXACT':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

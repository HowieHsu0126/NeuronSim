"""Compare source-clock observation bins on authenticated completed PFC events.

Native event ticks remain unchanged. Bins are observation summaries, not new
samples. Only existing training/calibration data enter bin-selection evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import h5py
import numpy as np

from .provenance import sha256_file, freeze_json
from .pfc_delay_analysis import READOUT_RTOL, READOUT_ATOL

CHUNK_EVENTS = 65536
BIN_TICKS = (80, 400)
SELECTED_CIRCUITS = (23, 63, 52, 86)
DURATION_TICKS = 400000
DT_MS = .0125
FIELDS = 17


def require(condition, message):
    if not condition:
        raise ValueError(message)


def directions(counts, x, y):
    """Reuse the existing cancellation rule; scientific weak-direction gate stays open."""
    counts, x, y = map(np.asarray, (counts, x, y))
    length = np.hypot(x, y)
    defined = (counts > 0) & (length > 64 * math.ulp(1.) * counts)
    angle = np.full(counts.shape, np.nan)
    angle[defined] = np.degrees(np.arctan2(y[defined], x[defined])) % 360
    return defined, angle


def control_boundaries(ticks, values, bin_ticks, duration_ticks):
    """Interior transitions mix bins only when they do not coincide with an edge."""
    ticks, values = np.asarray(ticks), np.asarray(values)
    require(ticks.ndim == 1 and ticks.dtype.kind in 'iu' and len(ticks) > 0
        and ticks[0] == 0 and np.all(np.diff(ticks) > 0)
        and ticks[-1] < duration_ticks and values.shape[0] == len(ticks)
        and values.ndim == 2 and np.isfinite(values).all(), 'OBSERVATION_CONTROL_DOMAIN_INVALID')
    changes = ticks[1:][np.any(values[1:] != values[:-1], axis=1)]
    return [dict(transition_tick=int(t), transition_ms=float(t * DT_MS),
        boundary_aligned=bool(t % bin_ticks == 0), mixed_bin=int(t // bin_ticks) if t % bin_ticks else None,
        mixed_start_tick=int(t // bin_ticks * bin_ticks) if t % bin_ticks else None,
        mixed_stop_tick=int((t // bin_ticks + 1) * bin_ticks) if t % bin_ticks else None) for t in changes]


def event_summary(chunks, core_units, *, duration_ticks=DURATION_TICKS):
    """Accumulate17 fields without a time x100000 dense matrix."""
    require(type(core_units) is int and core_units > 0 and core_units % 5 == 0,
        'OBSERVATION_COMPLETE_EI_UNIT_COUNT_REQUIRED')
    require(duration_ticks > 0 and duration_ticks % 400 == 0, 'OBSERVATION_COMPLETE_CLOCK_REQUIRED')
    n_e = core_units * 4 // 5
    mapping = np.r_[np.arange(n_e, dtype=np.int64) * 16 // n_e, np.full(core_units - n_e, 16)]
    fine = np.zeros((duration_ticks // 80, FIELDS), dtype=np.int64)
    x = np.zeros(len(fine)); y = np.zeros(len(fine))
    total_units = np.zeros(core_units, dtype=np.int64)
    late_units = np.zeros(core_units, dtype=np.int64)
    terminal, retained, terminal_ids = 0, 0, np.zeros(core_units, dtype=np.int64)
    cosine = np.cos(2 * np.pi * np.arange(n_e) / n_e)
    sine = np.sin(2 * np.pi * np.arange(n_e) / n_e)
    fine_coarse_event_identity = True
    for ticks, ids in chunks:
        ticks, ids = np.asarray(ticks), np.asarray(ids)
        require(ticks.shape == ids.shape and ticks.ndim == 1 and ticks.dtype.kind in 'iu'
            and ids.dtype.kind in 'iu' and np.all((ticks >= 0) & (ticks <= duration_ticks))
            and np.all((ids >= 0) & (ids < core_units)), 'OBSERVATION_INTEGER_EVENT_SUPPORT_INVALID')
        end = ticks == duration_ticks
        terminal += int(end.sum())
        terminal_ids += np.bincount(ids[end], minlength=core_units)
        ticks, ids = ticks[~end], ids[~end]
        retained += len(ticks)
        bins = ticks // 80
        fine_coarse_event_identity &= bool(np.array_equal(bins // 5, ticks // 400))
        fine += np.bincount(bins * FIELDS + mapping[ids], minlength=fine.size).reshape(fine.shape)
        total_units += np.bincount(ids, minlength=core_units)
        late = (ticks >= duration_ticks - 80000)
        late_units += np.bincount(ids[late], minlength=core_units)
        e = ids < n_e
        x += np.bincount(bins[e], weights=cosine[ids[e]], minlength=len(fine))
        y += np.bincount(bins[e], weights=sine[ids[e]], minlength=len(fine))
    coarse = fine.reshape(-1, 5, FIELDS).sum(axis=1)
    require(int(total_units.sum()) == int(fine.sum()) == int(coarse.sum()) == retained
        and fine_coarse_event_identity, 'OBSERVATION_INTEGER_CONSERVATION_FAILED')
    return dict(field_counts={80: fine, 400: coarse},
        vector_x={80: x, 400: x.reshape(-1, 5).sum(axis=1)},
        vector_y={80: y, 400: y.reshape(-1, 5).sum(axis=1)},
        unit_counts=total_units, late_unit_counts=late_units, terminal_events=terminal,
        terminal_unit_counts=terminal_ids, retained_events=retained,
        all_unit_integer_reaggregation_exact=fine_coarse_event_identity)


def validate_identity(identity, episode_id, role):
    c, b, k = (identity[name] for name in ('circuit_index', 'block_index', 'condition_index'))
    require(role in ('expanded', 'original') and c in SELECTED_CIRCUITS and b in (3, 4, 5)
        and type(k) is int and 0 <= k < 10 and identity['split'] == ('calibration' if c == 86 else 'train')
        and identity['episode_id'] == episode_id
        and episode_id == f'pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_{k:02d}',
        'OBSERVATION_SOURCE_IDENTITY_INVALID')
    stream_base = 210000000 if role == 'expanded' else 310000000
    require(identity['initial_state_seed'] == stream_base + c * 1000 + b
        and identity['background_seed'] == stream_base + 10000000 + c * 1000 + b,
        'OBSERVATION_DATASET_STREAM_IDENTITY_INVALID')


def authenticated_episode(directory, binding, role):
    """Consume summary-bound reports and event shards; never read unselected circuits."""
    report_path = directory / 'report.json'
    require(sha256_file(report_path) == binding[role + '_report_sha256'], 'OBSERVATION_READOUT_BINDING_CHANGED')
    report = json.loads(report_path.read_text())
    identity = report['identity']
    validate_identity(identity, binding['episode_id'], role)
    require(report['status'] == 'PASS_REFERENCE_SOURCE_QUALITY_READOUT'
        and report['source_qualification_granted'] is False and report['scientific_claim_qualified'] is False
        and report['production_episodes'] == 0 and report['dt_ms'] == DT_MS
        and report['duration_ticks'] == DURATION_TICKS
        and report['core_units'] == (100000 if role == 'expanded' else 2560), 'OBSERVATION_SOURCE_IDENTITY_INVALID')
    catalog_path = directory / 'export/dataset/dataset.json'
    require(sha256_file(catalog_path) == report['output_dataset_catalog_sha256'], 'OBSERVATION_CATALOG_CHANGED')
    catalog = json.loads(catalog_path.read_text()); config = catalog['binding']['resolved_source_config']
    require(catalog['complete_episodes'] == 1 and len(catalog['shards']) == 1
        and catalog['binding']['config_sha256'] == report['config_sha256']
        and config['benchmark_episode_candidate']['identity'] == identity,
        'OBSERVATION_CATALOG_IDENTITY_INVALID')
    shard = catalog['shards'][0]
    require(len(shard['episodes']) == 1 and shard['episodes'][0]['episode_id'] == identity['episode_id']
        and shard['episodes'][0]['event_count'] == report['events'], 'OBSERVATION_EVENT_DENOMINATOR_CHANGED')
    shard_relative = Path(shard['path'])
    require(not shard_relative.is_absolute() and '..' not in shard_relative.parts, 'OBSERVATION_SHARD_PATH_INVALID')
    shard_path = catalog_path.parent / shard_relative
    require(sha256_file(shard_path) == shard['sha256'], 'OBSERVATION_EVENT_SHARD_CHANGED')
    source = Path(report['source_archive'])
    require(sha256_file(source / 'delivery.json') == report['source_delivery_sha256'], 'OBSERVATION_SOURCE_CHANGED')
    source_delivery = json.loads((source / 'delivery.json').read_text())
    native_entries = [item for item in source_delivery['files'] if item['path'] == 'source_run/report.json']
    require(len(native_entries) == 1 and sha256_file(source / 'source_run/report.json') == native_entries[0]['sha256']
        and (report.get('source_report_sha256') is None or report['source_report_sha256'] == native_entries[0]['sha256']),
        'OBSERVATION_NATIVE_REPORT_CHANGED')
    native = json.loads((source / 'source_run/report.json').read_text())
    require(native['status'] == 'PFC_SOURCE_EXECUTION_COMPLETE' and native['source_execution_complete'] is True
        and native['resolved_config'] == config and native['config_sha256'] == report['config_sha256']
        and native['case_id'] == identity['episode_id'] and native['production_episodes'] == 0,
        'OBSERVATION_NATIVE_SOURCE_IDENTITY_CHANGED')
    circuit_id = shard['episodes'][0]['circuit_id']; circuit = catalog['circuits'][circuit_id]
    unit_path = catalog_path.parent / circuit['path']
    require(sha256_file(unit_path) == circuit['sha256'] and circuit['units'] == report['core_units'], 'OBSERVATION_UNIT_MAP_CHANGED')
    with h5py.File(unit_path, 'r') as h:
        n_e = report['core_units'] * 4 // 5
        expected_map = np.r_[np.arange(n_e) * 16 // n_e, np.full(report['core_units'] - n_e, 16)]
        require(np.array_equal(h['unit_ids'][:], np.arange(report['core_units']))
            and np.array_equal(h['field_index'][:], expected_map)
            and np.array_equal(h['populations'].asstr()[:], np.r_[np.full(n_e, 'E'), np.full(report['core_units'] - n_e, 'I')]),
            'OBSERVATION_SOURCE_POPULATION_MAP_INVALID')
    return report, shard_path


def process_episode(directory, binding, role):
    started = time.perf_counter(); report, shard = authenticated_episode(directory, binding, role)
    with h5py.File(shard, 'r') as h:
        group = h[report['identity']['episode_id']]
        metadata = json.loads(group.attrs['metadata_json'])
        require(metadata['episode_id'] == report['identity']['episode_id']
            and metadata['dt_ms'] == DT_MS and metadata['duration_ticks'] == DURATION_TICKS
            and metadata['seeds'] == dict(initial_rng_seed=report['identity']['initial_state_seed'],
                external_noise_seed=report['identity']['background_seed']), 'OBSERVATION_HDF5_IDENTITY_CHANGED')
        require(group['spike_ticks'].shape == group['spike_unit_indices'].shape == (report['events'],),
            'OBSERVATION_HDF5_EVENT_DENOMINATOR_CHANGED')
        def chunks():
            for first in range(0, report['events'], CHUNK_EVENTS):
                stop = min(first + CHUNK_EVENTS, report['events'])
                yield group['spike_ticks'][first:stop], group['spike_unit_indices'][first:stop]
        values = event_summary(chunks(), report['core_units'])
        controls = group['control_ticks'][:], group['control_values'][:]
    require(values['terminal_events'] == report['terminal_tick_events']
        and values['retained_events'] + values['terminal_events'] == report['events'], 'OBSERVATION_TERMINAL_EVENTS_CHANGED')
    # Authenticate and compare the existing5ms measurements; no new export is built.
    path = directory / 'unit_population_statistics.npz'
    require(sha256_file(path) == report['outputs'][path.name]['sha256'], 'OBSERVATION_EXISTING_COUNTS_CHANGED')
    with np.load(path, allow_pickle=False) as old:
        require(np.array_equal(values['field_counts'][400], old['population_field_counts'])
            and np.array_equal(values['unit_counts'], old['phase_unit_counts'].sum(axis=0)),
            'OBSERVATION_EXISTING_INTEGER_COUNTS_DIFFER')
    rows, boundaries, arrays = [], [], {}
    n_e = report['core_units'] * 4 // 5
    sizes = np.r_[np.bincount(np.arange(n_e) * 16 // n_e, minlength=16), report['core_units'] - n_e]
    late_units = values['late_unit_counts']; late_e = int(late_units[:n_e].sum())
    late_x = math.fsum(int(n) * math.cos(2 * math.pi * unit / n_e) for unit, n in enumerate(late_units[:n_e]) if n)
    late_y = math.fsum(int(n) * math.sin(2 * math.pi * unit / n_e) for unit, n in enumerate(late_units[:n_e]) if n)
    late_defined, late_angle = directions(np.array([late_e]), np.array([late_x]), np.array([late_y]))
    for width in BIN_TICKS:
        fields = values['field_counts'][width]; counts = fields[:, :16].sum(axis=1)
        x, y = values['vector_x'][width], values['vector_y'][width]
        defined, angle = directions(counts, x, y)
        boundary = control_boundaries(*controls, width, DURATION_TICKS)
        for item in boundary:
            boundaries.append(dict(episode_id=report['identity']['episode_id'], source_size=report['core_units'], bin_ticks=width, **item))
        rates = fields / sizes / (width * DT_MS * .001)
        temporal_covariance = np.cov(rates, rowvar=False, ddof=1)
        late = slice(320000 // width, 400000 // width)
        rows.append(dict(episode_id=report['identity']['episode_id'], circuit_index=report['identity']['circuit_index'],
            block_index=report['identity']['block_index'], condition_id=report['condition']['condition_id'], split=report['identity']['split'],
            source_size=report['core_units'], bin_ticks=width, bin_ms=width * DT_MS, bins=len(fields),
            native_event_ticks_preserved=True, retained_events=values['retained_events'], terminal_events=values['terminal_events'],
            all_unit_integer_reaggregation_exact=values['all_unit_integer_reaggregation_exact'], existing5ms_counts_exact=True,
            current_transitions=len(boundary), unaligned_transitions=sum(not x['boundary_aligned'] for x in boundary),
            silent_E_bins=int(np.count_nonzero(counts == 0)), direction_defined_bins=int(defined.sum()),
            late_silent_E_bins=int(np.count_nonzero(counts[late] == 0)), late_direction_defined_bins=int(defined[late].sum()),
            late_E_events=late_e, late_E_rate_Hz=late_e / n_e,
            late_phase_direction_defined=bool(late_defined[0]), late_phase_angle_deg=float(late_angle[0]) if late_defined[0] else None,
            direction_scientific_eligibility='UNQUALIFIED_NO_FROZEN_MINIMUM_ACTIVITY_OR_RESULTANT_GATE',
            covariance_trace_rate_Hz2=float(np.trace(temporal_covariance)),
            grouped_counts_bytes=fields.nbytes, grouped_vector_bytes=x.nbytes + y.nbytes,
            per_episode_dense_allunit_values_avoided=len(fields) * report['core_units']))
        arrays[f'field_counts_{width}'] = fields
        arrays[f'vector_x_{width}'] = x; arrays[f'vector_y_{width}'] = y
        arrays[f'angle_{width}'] = angle; arrays[f'direction_defined_{width}'] = defined
        arrays[f'temporal_covariance_{width}'] = temporal_covariance
    elapsed = time.perf_counter() - started
    for row in rows:
        row['authenticated_read_and_joint_aggregation_seconds'] = elapsed
    return rows, boundaries, arrays


def write_rows(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size-summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--representative-only',action='store_true')
    args=parser.parse_args(argv)
    require(not args.output.exists(),'OBSERVATION_OUTPUT_EXISTS')
    summary=json.loads(args.size_summary.read_text())
    require(summary['status']=='COMPLETE120_DESCRIPTIVE_SIZE_CALIBRATION'
        and summary['circuits']==list(SELECTED_CIRCUITS) and len(summary['bindings'])==120
        and summary['original_size_episodes']==120 and summary['expanded_size_episodes_reused']==120,
        'OBSERVATION_COMPLETE_TRAINING_CALIBRATION_SUMMARY_REQUIRED')
    selected=summary['bindings'][:1] if args.representative_only else summary['bindings']
    args.output.mkdir(parents=True);(args.output/'episodes').mkdir()
    all_rows,boundaries=[],[];started=time.perf_counter();events=0
    try:
        for binding in selected:
            for role in ('expanded','original'):
                rows, transitions, arrays=process_episode(Path(binding[role+'_readout']),binding,role)
                all_rows+=rows;boundaries+=transitions;events+=rows[0]['retained_events']+rows[0]['terminal_events']
                np.savez_compressed(args.output/'episodes'/(binding['episode_id']+'_'+role+'.npz'),**arrays)
                print(json.dumps(dict(episode=binding['episode_id'],role=role,seconds=rows[0]['authenticated_read_and_joint_aggregation_seconds'])),flush=True)
        write_rows(args.output/'candidate_bins.csv',all_rows);write_rows(args.output/'control_boundaries.csv',boundaries)
        fine=[x for x in all_rows if x['bin_ticks']==80]
        coarse=[x for x in all_rows if x['bin_ticks']==400]
        report=dict(status='DESCRIBED_OBSERVATION_RESOLUTION_COHORT',complete_source_episodes=len(fine),
            source_sizes=[100000,2560],circuits=list(SELECTED_CIRCUITS),blocks=[3,4,5],
            training_calibration_only=True,bin_candidates_ticks=list(BIN_TICKS),integer_events_read=events,
            representative_only=args.representative_only,
            boundary_criterion_satisfied=all(x['unaligned_transitions']==0 and x['all_unit_integer_reaggregation_exact'] for x in fine),
            recommended_bin_ticks=80 if all(x['unaligned_transitions']==0 for x in fine) else None,
            coarse_unaligned_transitions=sum(x['unaligned_transitions'] for x in coarse),
            criterion='preserve source1ms current transitions with exact integer event conservation; native event ticks remain stored',
            observation_bin_frozen=False,scientific_accuracy_qualified=False,direction_scientific_eligibility_qualified=False,
            source_qualification_granted=False,scaling_qualified=False,production_episodes=0,new_source_simulations=0,
            temporal_covariance_unit='within_episode time bins; not independent histories',
            runtime_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            summary=dict(path=str(args.size_summary.resolve()),sha256=sha256_file(args.size_summary)),
            implementation_sha256=sha256_file(Path(__file__)),
            outputs={name:sha256_file(args.output/name) for name in ('candidate_bins.csv','control_boundaries.csv')})
        freeze_json(args.output/'report.json',report);print(json.dumps(report))
    except BaseException as error:
        freeze_json(args.output/'failure.json',dict(error=repr(error),completed_cells=len(all_rows)//2,
            source_qualification_granted=False,production_episodes=0))
        raise


if __name__=='__main__':
    main()

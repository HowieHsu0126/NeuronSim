"""Consecutive source-clock PFC spike readouts from already completed episodes.

All conditions and empty bins are retained. These descriptive readouts neither
define a memory-success threshold nor reconstruct unrecorded state trajectories.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_scale_analysis as scale_analysis

from .pfc_scale_analysis import (
    _source_bound_file, _source_header, _source_manifest, _source_registration,
    _source_values, scalar_readout,
)
from .pfc_delay_analysis import PHASES, CONDITIONS, READOUT_RTOL, READOUT_ATOL, read, require, write_json, write_rows
from .provenance import sha256_file


RATE_FIELDS = ('excitatory_rate_hz', 'inhibitory_rate_hz', 'cue_region_rate_hz',
               'noncue_region_rate_hz', 'cue_selectivity_hz', 'all_unit_rate_hz')
COUNT_FIELDS = ('excitatory_spikes', 'inhibitory_spikes')
MOMENT_FIELDS = ('vector_x_spikes', 'vector_y_spikes')


def lagged_angular_displacement(rows, phase_edges_ticks, *, bin_ticks, lag_ticks):
    """Describe principal circular increments of saved bin vectors within phases.

    Overlapping temporal pairs are not independent simulation units. Silent or
    cancelled vectors remain undefined. No unwrapped path or diffusion fit is
    inferred from these finite-bin directions.
    """
    require(type(bin_ticks) is int and bin_ticks > 0
            and all(type(lag) is int and lag > 0 and lag % bin_ticks == 0 for lag in lag_ticks)
            and len(set(lag_ticks)) == len(lag_ticks), 'ANGULAR_EXACT_POSITIVE_LAGS_REQUIRED')
    edges = np.asarray(phase_edges_ticks)
    require(edges.ndim == 1 and edges.dtype.kind in 'iu' and len(edges) > 1
            and edges[0] == 0 and np.all(np.diff(edges) > 0), 'ANGULAR_PHASE_DOMAIN_REQUIRED')
    require(len(rows) > 0 and int(rows[0]['start_tick']) == 0
            and int(rows[-1]['stop_tick']) == int(edges[-1]), 'ANGULAR_COMPLETE_BIN_DOMAIN_REQUIRED')
    directions = []
    for index, row in enumerate(rows):
        start, stop = int(row['start_tick']), int(row['stop_tick'])
        require(start == index * bin_ticks and stop == (index + 1) * bin_ticks,
                'ANGULAR_CONSECUTIVE_BINS_REQUIRED')
        total = int(row['excitatory_spikes'])
        x, y = (float(row[name]) for name in MOMENT_FIELDS)
        require(total >= 0 and math.isfinite(x) and math.isfinite(y)
                and math.hypot(x, y) <= total + 64 * math.ulp(1.) * max(1, total),
                'ANGULAR_VECTOR_SUPPORT_INVALID')
        defined = total > 0 and math.hypot(x, y) > 64 * math.ulp(1.) * total
        flag = row['direction_defined']
        require(flag in (True, False, 'True', 'False', 'true', 'false', '1', '0')
                and (flag in (True, 'True', 'true', '1')) == defined,
                'ANGULAR_DIRECTION_FLAG_CHANGED')
        angle = row['decoded_angle_deg']
        if defined:
            angle = float(angle)
            expected = math.degrees(math.atan2(y, x)) % 360
            require(math.isfinite(angle) and abs((angle - expected + 180) % 360 - 180) <= 1e-9,
                    'ANGULAR_SAVED_DIRECTION_CHANGED')
        else:
            require(angle in (None, ''), 'ANGULAR_UNDEFINED_DIRECTION_CHANGED')
            angle = None
        directions.append(angle)
    results = []
    for phase, (begin, end) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [index for index, row in enumerate(rows)
                    if int(row['start_tick']) >= begin and int(row['stop_tick']) <= end]
        covered = len(selected) * bin_ticks
        for lag in lag_ticks:
            offset = lag // bin_ticks
            pairs = list(zip(selected[:-offset], selected[offset:]))
            require(all((right - left) * bin_ticks == lag for left, right in pairs),
                    'ANGULAR_EXACT_LAG_PAIR_REQUIRED')
            increments = [((directions[right] - directions[left] + 180) % 360 - 180)
                          for left, right in pairs
                          if directions[left] is not None and directions[right] is not None]
            endpoints = {index for pair in pairs for index in pair}
            complete = len(pairs) > 0 and len(increments) == len(pairs)
            results.append(dict(phase_index=phase, phase_start_tick=int(begin), phase_stop_tick=int(end),
                bin_ticks=bin_ticks, lag_ticks=lag, phase_whole_bins=len(selected),
                phase_covered_ticks=covered, phase_uncovered_ticks=int(end-begin)-covered,
                candidate_pairs=len(pairs), defined_pairs=len(increments),
                undefined_pairs=len(pairs)-len(increments), unique_endpoint_bins=len(endpoints),
                undefined_unique_endpoint_bins=sum(directions[index] is None for index in endpoints),
                mean_signed_displacement_deg=math.fsum(increments)/len(increments) if complete else None,
                mean_squared_displacement_deg2=math.fsum(value*value for value in increments)/len(increments) if complete else None,
                mean_cosine_increment=math.fsum(math.cos(math.radians(value)) for value in increments)/len(increments) if complete else None,
                mean_sine_increment=math.fsum(math.sin(math.radians(value)) for value in increments)/len(increments) if complete else None,
                missing_reason='' if complete else 'no_phase_contained_pairs' if not pairs else 'undefined_endpoint_pairs',
                sample_axis='overlapping_temporal_pairs_within_one_episode',
                angular_convention='principal_displacement_in_minus180_inclusive_to180_exclusive'))
    return results


def delay_summary(rows, delay_start_ms):
    selected = [row for row in rows if float(row['start_ms']) >= delay_start_ms]
    require(selected, 'complete post-cue readout interval required')
    selectivity = [float(row['cue_selectivity_hz']) for row in selected]
    errors = [abs(float(row['circular_cue_error_deg'])) for row in selected
              if row['circular_cue_error_deg'] not in (None, '')]
    require(all(np.isfinite(value) for value in selectivity+errors), 'finite descriptive readouts required')
    return dict(delay_start_ms=delay_start_ms, delay_stop_ms=float(selected[-1]['stop_ms']),
        delay_bins=len(selected), excitatory_spikes=sum(int(row['excitatory_spikes']) for row in selected),
        inhibitory_spikes=sum(int(row['inhibitory_spikes']) for row in selected),
        cue_selectivity_min_hz=min(selectivity), cue_selectivity_max_hz=max(selectivity),
        cue_selectivity_mean_hz=math.fsum(selectivity)/len(selectivity),
        direction_defined_bins=len(errors), max_abs_circular_cue_error_deg=max(errors) if errors else None)


def summarize_existing(directory, output):
    directory, output = Path(directory), Path(output)
    report = read(directory/'analysis.json')
    expected_cases = report.get('registered_source_cases', 9)
    expected_bins = round(report.get('duration_ms', 5000.)/report.get('bin_ms', 5.))
    require(type(expected_cases) is int and expected_cases >= 9
            and report['status'] == 'COMPLETE_SOURCE_TEMPORAL_READOUTS'
            and len(report['completed_cases']) == expected_cases
            and report['bins'] == expected_cases*expected_bins,
            'complete registered temporal analysis required')
    summaries = []
    for case in report['completed_cases']:
        path = directory/case['path']
        require(Path(case['path']).name == case['path'] and sha256_file(path) == case['sha256'],
                'temporal CSV identity changed')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == case['bins'] and all(row['case_id'] == case['case_id'] for row in rows),
                'complete temporal case rows required')
        info = {key: case[key] for key in ('case_id', 'source_rng_seed', 'condition', 'stimulus_strength_nA')}
        summaries.append(dict(info, **delay_summary(rows, PHASES['early_delay'][0])))
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/'delay_summary.csv', summaries)
    write_json(output/'summary.json', dict(source_analysis=str(directory/'analysis.json'),
        source_analysis_sha256=sha256_file(directory/'analysis.json'), cases=summaries,
        source_csv_sha256={case['path']:case['sha256'] for case in report['completed_cases']},
        implementation_sha256=sha256_file(Path(__file__)),
        delay_summary_sha256=sha256_file(output/'delay_summary.csv'),
        scope='descriptive trajectories from registered nominal cue end; no memory-success threshold or significance test',
        delay_boundary_semantics=report.get('delay_boundary_semantics',
            'registered nominal300ms boundary; not verified actual current removal'),
        current_observation_status=report.get('current_observation_status', 'NOT_ESTABLISHED_BY_TEMPORAL_ANALYSIS'),
        coupled_core_units=report.get('coupled_core_units'),
        source_rng_seeds=report.get('source_rng_seeds'),
        working_memory_qualified=False, scientific_fidelity_qualified=False))


def temporal_rows(ticks, ids, *, n_e, n_i, dt_ms, bin_ms, duration_ms):
    ticks, ids = np.asarray(ticks), np.asarray(ids)
    require(ticks.ndim == ids.ndim == 1 and ticks.shape == ids.shape
            and ticks.dtype.kind in 'iu' and ids.dtype.kind in 'iu', 'integer source event vectors required')
    require(n_e > 0 and n_i > 0 and all(np.isfinite(x) and x > 0 for x in (dt_ms, bin_ms, duration_ms)),
            'positive populations and finite source clocks required')
    bin_ticks, duration_ticks = round(bin_ms/dt_ms), round(duration_ms/dt_ms)
    require(bin_ticks > 0 and duration_ticks > 0 and duration_ticks % bin_ticks == 0
            and np.isclose(bin_ticks*dt_ms, bin_ms, rtol=0, atol=1e-9)
            and np.isclose(duration_ticks*dt_ms, duration_ms, rtol=0, atol=1e-9),
            'complete equal-width source-clock bins required')
    require(not np.any(ticks[1:] < ticks[:-1]) and np.all((ticks >= 0) & (ticks < duration_ticks))
            and np.all((ids >= 0) & (ids < n_e+n_i)), 'ordered events in the complete half-open source domain required')
    edges = np.arange(0, duration_ticks+1, bin_ticks, dtype=np.int64)
    offsets = np.searchsorted(ticks, edges, side='left')
    rows = []
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        rows.append(dict(bin_index=index, start_ms=float(edges[index]*dt_ms),
            stop_ms=float(edges[index+1]*dt_ms), start_tick=int(edges[index]), stop_tick=int(edges[index+1]),
            **scalar_readout(ids[start:stop], n_e, n_i, bin_ms)))
    require(sum(row['excitatory_spikes']+row['inhibitory_spikes'] for row in rows) == len(ids),
            'temporal bins must conserve every retained event')
    return rows


def compare_phase(rows, reference):
    """Compare additive moments and duration-weighted rates; never average angles."""
    start, stop = float(reference['start_ms']), float(reference['stop_ms'])
    selected = [row for row in rows if row['start_ms'] >= start and row['stop_ms'] <= stop]
    require(selected and selected[0]['start_ms'] == start and selected[-1]['stop_ms'] == stop
            and all(left['stop_ms'] == right['start_ms'] for left, right in zip(selected[:-1], selected[1:])),
            'reference phase must be covered by consecutive complete bins')
    values = {name: sum(row[name] for row in selected) for name in COUNT_FIELDS}
    values.update({name: math.fsum(row[name] for row in selected) for name in MOMENT_FIELDS})
    values.update({name: math.fsum(row[name]*(row['stop_ms']-row['start_ms']) for row in selected)/(stop-start)
                   for name in RATE_FIELDS})
    require(len({row['cue_units'] for row in selected}) == 1, 'source cue geometry changed within an episode')
    values['cue_units'] = selected[0]['cue_units']
    comparisons = []
    for name, actual in values.items():
        expected = float(reference[name])
        exact = name in (*COUNT_FIELDS, 'cue_units')
        passed = actual == expected if exact else abs(actual-expected) <= READOUT_ATOL+READOUT_RTOL*abs(expected)
        require(passed, 'temporal aggregation differs from completed phase readout: '+name)
        comparisons.append(dict(phase=reference['phase'], metric=name, expected=expected,
                                actual=actual, absolute_difference=abs(actual-expected), exact=exact))
    return comparisons


def analyze(config_path, source_runs, reference, output):
    config_path, reference, output = map(Path, (config_path, reference, output))
    if output.exists():
        raise FileExistsError(output)
    config, registered, seeds, dimensions = _source_registration(config_path)
    config_sha = sha256_file(config_path)
    receipt = read(reference/'delivery.json')
    require(receipt.get('status') == 'ARCHIVED_PFC_SOURCE_READOUTS' and receipt.get('exit_code') == 0,
            'completed source phase readout delivery required')
    manifest = _source_manifest(receipt)
    checked = {name: _source_bound_file(reference, manifest, name)
               for name in ('config.yaml', 'input_origins.json', 'analysis/responses.csv')}
    require(sha256_file(checked['config.yaml']) == config_sha, 'continuous and phase source configurations differ')
    origins = read(checked['input_origins.json'])
    require(len(origins) == len(registered) and {row['case_id'] for row in origins} == set(registered),
            'complete original registered source bindings required')
    origin_by_case = {row['case_id']: row for row in origins}
    with checked['analysis/responses.csv'].open(newline='') as stream:
        previous = list(csv.DictReader(stream))
    expected_phases = {(case, phase) for case in registered for phase in PHASES}
    require(len(previous) == len(expected_phases)
            and {(row['case_id'], row['phase']) for row in previous} == expected_phases,
            'complete previous phase denominator required')
    headers = [_source_header(path, config, registered, config_sha) for path in source_runs]
    require(len(headers) == len(registered) and all(status['status'] == 'COMPLETE' for status, _ in headers)
            and {status['case_id'] for status, _ in headers} == set(registered),
            'all registered source episodes must be complete')
    output.mkdir(parents=True)
    record = dict(status='READING_COMPLETE_SOURCE_EVENTS', config_sha256=config_sha,
        reference_delivery_sha256=sha256_file(reference/'delivery.json'), source_rng_seeds=seeds,
        dt_ms=config['dt_ms'], bin_ms=config['readout_bin_ms'], duration_ms=config['duration_ms'],
        coupled_core_units=dimensions['coupled_core_units'], registered_source_cases=len(registered), completed_cases=[],
        scientific_fidelity_qualified=False, working_memory_qualified=False, production_episodes=0,
        observation='all-unit spike events; source-defined excitatory preference angles',
        unrecorded_trajectories=['voltage', 'synaptic_state'],
        implementation_sha256=sha256_file(Path(__file__)),
        source_reader_sha256=sha256_file(Path(__file__).with_name('pfc_scale_analysis.py')))
    write_json(output/'analysis.json', record)
    comparisons = []
    for status, header in headers:
        started = time.perf_counter()
        case = status['case_id']; seed, condition = registered[case]
        require(status['delivery_sha256'] == origin_by_case[case]['delivery_sha256'],
                'source delivery differs from the completed phase analysis')
        raw, _, _ = _source_values(header, config, registered, dimensions, config_sha)
        rows = temporal_rows(raw['spike_ticks'], raw['spike_unit_indices'],
            n_e=dimensions['n_e'], n_i=dimensions['n_i'], dt_ms=config['dt_ms'],
            bin_ms=config['readout_bin_ms'], duration_ms=config['duration_ms'])
        for phase in (row for row in previous if row['case_id'] == case):
            comparisons.extend(dict(case_id=case, **row) for row in compare_phase(rows, phase))
        info = dict(case_id=case, source_rng_seed=seed, condition=condition,
                    stimulus_strength_nA=CONDITIONS[condition], excitatory_neurons=dimensions['n_e'],
                    inhibitory_neurons=dimensions['n_i'])
        path = output/(case+'.csv')
        write_rows(path, [dict(info, **row) for row in rows])
        record['completed_cases'].append(dict(info, bins=len(rows), events=len(raw['spike_ticks']),
            source_binding=header['binding'], path=path.name, sha256=sha256_file(path),
            elapsed_seconds=time.perf_counter()-started))
        write_json(output/'analysis.json', record)
        print(f"Completed temporal readout: {case}, {len(rows)} bins", flush=True)
        del raw, rows
    write_rows(output/'phase_comparisons.csv', comparisons)
    record.update(status='COMPLETE_SOURCE_TEMPORAL_READOUTS', bins=sum(row['bins'] for row in record['completed_cases']),
                  phase_linear_statistic_comparisons=len(comparisons),
                  phase_comparisons_sha256=sha256_file(output/'phase_comparisons.csv'))
    write_json(output/'analysis.json', record)
    return record


def analyze_native_reference(archive, output):
    """Read the complete legacy reference archive; never execute its simulator."""
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='CHECKING_NATIVE_REFERENCE', completed_cases=[],
        source_format='legacy_native_brian2', source_reference_archive=str(archive),
        scientific_fidelity_qualified=False, working_memory_qualified=False,
        full_scale_qualified=False, production_episodes=0, new_source_episodes=0,
        confidence_intervals=False, p_values=False, q_values=False,
        current_observation_status='UNAVAILABLE_LEGACY_ARCHIVE',
        control_scope='archived prescribed cue/distractor commands; actual delivered currents unrecorded',
        delay_boundary_semantics='registered nominal300ms boundary; not verified actual current removal',
        observation='all-unit spike events; source-defined excitatory preference angles',
        unrecorded_trajectories=['voltage', 'synaptic_state', 'actual_delivered_current'],
        implementation_sha256=sha256_file(Path(__file__)),
        source_reader_sha256=sha256_file(Path(scale_analysis.__file__)))
    consumed = {}
    try:
        delivery = read(archive/'delivery.json')
        require(delivery.get('status') == 'ARCHIVED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC'
                and delivery.get('complete_source_episodes') == 9
                and delivery.get('units') == 2560 and delivery.get('duration_ms') == 5000
                and delivery.get('source_rng_seeds') == list(scale_analysis.SOURCE_SEEDS)
                and delivery.get('source_revision') == scale_analysis.SOURCE_REVISION
                and delivery.get('source_entry_sha256') == scale_analysis.SOURCE_ENTRY_SHA256
                and delivery.get('production_episodes') == 0
                and delivery.get('scientific_fidelity_qualified') is False
                and delivery.get('full_scale_qualified') is False
                and delivery.get('working_memory_qualified') is False,
                'complete unqualified reference-size delivery required')
        manifest = _source_manifest(delivery)

        def bound(name):
            path = Path(name)
            require(not path.is_absolute() and '..' not in path.parts,
                    'native reference manifest path escaped archive')
            name = path.as_posix()
            if name not in consumed:
                consumed[name] = _source_bound_file(archive, manifest, name)
            return consumed[name]

        # The archive also preserves an interrupted native/ prefix.
        native_names = [name for name, row in manifest.items()
                        if name.endswith('/report.json')
                        and row['sha256'] == delivery.get('native_report_sha256')]
        require(len(native_names) == 1, 'unique complete native report binding required')
        native_name = native_names[0]
        native_root = bound(native_name).parent
        paths = dict(native=bound(native_name),
            factorized=bound('factorized/report.json'),
            fresh=bound('factorized_replay/report.json'),
            comparison=bound('source_comparison.json'),
            replay=bound('factorized_replay.json'))
        design_path = bound('design.json')
        config_paths = [
            bound('Libs/configs/neural_population_pfc_reference_size.yaml'),
            bound('Libs/configs/neural_population_pfc_reference_size_factorized.yaml')]
        configs = [yaml.safe_load(path.read_text()) for path in config_paths]
        design = read(design_path)
        loaded, registered, _ = scale_analysis.preflight(paths, design, configs,
            comparison_implementation=bound('Libs/population_dynamics/pfc_source_comparison.py'),
            replay_implementation=bound('Libs/population_dynamics/pfc_reference_replay.py'))
        config, report = configs[0], loaded['native']
        require(loaded['comparison']['design_sha256'] == sha256_file(design_path)
                and sha256_file(paths['comparison']) == delivery.get('source_comparison_sha256')
                and sha256_file(paths['replay']) == delivery.get('fresh_replay_sha256'),
                'native comparison or design delivery binding changed')

        phase_path = bound('analysis/analysis.json')
        replay_path = bound('analysis/readout_replay.json')
        response_path = bound('analysis/responses.csv')
        phase, replay = read(phase_path), read(replay_path)
        native_sha = sha256_file(paths['native'])
        source_code = bound('Libs/population_dynamics/pfc_scale_analysis.py')
        require(sha256_file(phase_path) == delivery.get('analysis_sha256')
                and phase.get('status') == 'DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC'
                and phase.get('native_report_sha256') == native_sha
                and phase.get('configuration') == config
                and phase.get('complete_source_episodes') == 9 and phase.get('units') == 2560
                and phase.get('source_rng_seeds') == list(scale_analysis.SOURCE_SEEDS)
                and phase.get('phases_ms') == PHASES
                and phase.get('source_comparison_sha256') == sha256_file(paths['comparison'])
                and phase.get('fresh_replay_sha256') == sha256_file(paths['replay'])
                and phase.get('design_sha256') == sha256_file(design_path)
                and phase.get('readout_replay_sha256') == sha256_file(replay_path)
                and phase.get('implementation_sha256') == sha256_file(source_code)
                and phase.get('scientific_fidelity_qualified') is False
                and phase.get('working_memory_qualified') is False
                and phase.get('production_episodes') == 0,
                'reference phase analysis provenance changed')
        require(replay.get('status') == 'PASS' and replay.get('complete_episodes') == 9
                and replay.get('units') == 2560 and replay.get('source_rng_blocks') == 3
                and replay.get('native_report_sha256') == native_sha
                and replay.get('rtol') == READOUT_RTOL and replay.get('atol') == READOUT_ATOL
                and replay.get('implementation_sha256') == sha256_file(source_code)
                and replay.get('artifacts') == phase.get('artifacts')
                and replay.get('scientific_fidelity_qualified') is False
                and replay.get('phase_unit_count_values_checked') == 9*len(PHASES)*2560
                and phase.get('phase_unit_count_values_verified') == replay['phase_unit_count_values_checked']
                and phase.get('scalar_readout_values_verified') == replay.get('scalar_readout_values_checked')
                and replay['artifacts'].get('responses.csv') == sha256_file(response_path),
                'reference phase readout replay binding changed')
        with response_path.open(newline='') as stream:
            previous = list(csv.DictReader(stream))
        expected_phases = {(case, phase_name) for case in registered for phase_name in PHASES}
        require(len(previous) == len(expected_phases)
                and {(row['case_id'], row['phase']) for row in previous} == expected_phases,
                'complete native reference phase coverage required')

        prefix = native_root.relative_to(archive)
        catalog = read(bound((prefix/'dataset/dataset.json').as_posix()))
        for circuit in catalog['circuits'].values():
            bound((prefix/'dataset'/circuit['path']).as_posix())
        for shard in catalog['shards']:
            bound((prefix/'dataset'/shard['path']).as_posix())
        for case in report['cases']:
            require(Path(case['raw_file']).name == case['raw_file'],
                    'native raw path escaped complete source root')
            bound((prefix/case['raw_file']).as_posix())
        dataset, identity = scale_analysis.dataset_binding(
            native_root, report, config, sha256_file(config_paths[0]), registered)
        n_e, n_i = config['excitatory_neurons'], config['inhibitory_neurons']
        dt_ms, bin_ms, duration_ms = config['dt_ms'], config['readout_bin_ms'], config['duration_ms']
        duration_ticks, bin_ticks = round(duration_ms/dt_ms), round(bin_ms/dt_ms)
        record.update(status='READING_COMPLETE_SOURCE_EVENTS',
            native_report_path=native_name, native_report_sha256=native_sha,
            reference_delivery_sha256=sha256_file(archive/'delivery.json'),
            reference_readout_replay_sha256=sha256_file(replay_path),
            config_sha256=sha256_file(config_paths[0]),
            source_rng_seeds=list(scale_analysis.SOURCE_SEEDS), source_rng_scope='joint initial/background blocks',
            dt_ms=dt_ms, bin_ms=bin_ms, duration_ms=duration_ms, coupled_core_units=n_e+n_i,
            source_backend='native_brian2', source_versions={
                key: report[key] for key in ('brian2_version', 'numpy_version')},
            terminal_tick_events_excluded=0, source_episodes_reused=9)
        write_json(output/'analysis.json', record)
        comparisons = []
        indexed = {case['case_id']: case for case in report['cases']}
        for case_id, (seed, condition) in registered.items():
            started = time.perf_counter()
            case = indexed[case_id]
            raw = scale_analysis.raw_values(native_root, case)
            dense = scale_analysis.archive_window(dataset, case, raw, seed, condition, identity)
            # Legacy raw archives retain terminal-tick events outside [0, duration).
            retained = raw['spike_ticks'] < duration_ticks
            excluded = int(np.count_nonzero(~retained))
            ticks, ids = raw['spike_ticks'][retained], raw['spike_unit_indices'][retained]
            flat = (ticks//bin_ticks)*(n_e+n_i)+ids
            expected = np.bincount(flat, minlength=dense.size).reshape(dense.shape)
            require(np.array_equal(expected, dense),
                    'native continuous all-unit archive counts differ from raw events')
            rows = temporal_rows(ticks, ids, n_e=n_e, n_i=n_i, dt_ms=dt_ms,
                                 bin_ms=bin_ms, duration_ms=duration_ms)
            for old in (row for row in previous if row['case_id'] == case_id):
                comparisons.extend(dict(case_id=case_id, **row) for row in compare_phase(rows, old))
            info = dict(case_id=case_id, source_rng_seed=seed, condition=condition,
                stimulus_strength_nA=CONDITIONS[condition], excitatory_neurons=n_e, inhibitory_neurons=n_i)
            path = output/(case_id+'.csv')
            write_rows(path, [dict(info, **row) for row in rows])
            record['completed_cases'].append(dict(info, bins=len(rows), events=len(ids),
                archived_events=len(raw['spike_ticks']), terminal_tick_events_excluded=excluded,
                source_binding=dict(native_report_path=native_name, native_report_sha256=native_sha,
                    raw_path=(prefix/case['raw_file']).as_posix(), raw_sha256=case['raw_sha256']),
                path=path.name, sha256=sha256_file(path), elapsed_seconds=time.perf_counter()-started))
            record['terminal_tick_events_excluded'] += excluded
            write_json(output/'analysis.json', record)
            print(f"Completed native reference temporal readout: {case_id}, {len(rows)} bins", flush=True)
            del raw, dense, expected, rows, ticks, ids, flat
        write_rows(output/'phase_comparisons.csv', comparisons)
        bins = sum(row['bins'] for row in record['completed_cases'])
        require(len(record['completed_cases']) == 9 and bins == 9000,
                'complete native reference temporal coverage required')
        record.update(status='COMPLETE_SOURCE_TEMPORAL_READOUTS', bins=bins,
            phase_linear_statistic_comparisons=len(comparisons),
            phase_comparisons_sha256=sha256_file(output/'phase_comparisons.csv'))
    except Exception as error:
        record.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['consumed_manifest_files'] = {
            name: manifest[name]['sha256'] for name in consumed}
        write_json(output/'analysis.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'reference-readouts', 'summary-of', 'native-reference-archive'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, action='append')
    args = parser.parse_args(argv)
    if args.native_reference_archive is not None:
        require(args.config is None and args.source_run is None and args.reference_readouts is None
                and args.summary_of is None, 'native reference archive mode cannot combine other input modes')
        analyze_native_reference(args.native_reference_archive, args.output)
    elif args.summary_of is not None:
        require(args.config is None and args.source_run is None and args.reference_readouts is None,
                'summary consumes only completed temporal readouts')
        summarize_existing(args.summary_of, args.output)
    else:
        require(args.config is not None and args.source_run and args.reference_readouts is not None,
                'config, completed sources and reference readouts required')
        analyze(args.config, args.source_run, args.reference_readouts, args.output)


if __name__ == '__main__':
    main()

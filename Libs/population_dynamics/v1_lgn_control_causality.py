"""Compare source half-contrast rates through representative execution or full archives."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import importlib
import inspect
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_configured_inputs import delivered_file
from .v1_lgn_runtime import load_prepared_runtime, probe_runtime
from .v1_lgn_source import read_source_cells
from .v1_nest_input_replay import bound_json

SOURCE_UNITS = 17400
SOURCE_MODELS = 14
UNCHANGED_MOVIE_FRAMES = 501
RATE_SAMPLES = 3001


def validate_config_pair(baseline: dict, control: dict) -> None:
    """Only coverage and the prescribed half-contrast intervention may differ."""
    if (baseline.get('scope') != 'development_source_runtime_qualification'
            or baseline.get('unit_coverage') != 'all_source_units'):
        raise ValueError('SOURCE_CONTROL_FULL_DEVELOPMENT_BASELINE_REQUIRED')
    expected = dict(baseline, unit_coverage='model_representatives')
    expected['stimulus'] = dict(baseline['stimulus'])
    expected['stimulus']['contrast_percent'] /= 2
    if control != expected:
        raise ValueError('SOURCE_CONTROL_ONLY_HALF_CONTRAST_AND_COVERAGE_MAY_DIFFER')


def validate_movie_witness(movie: dict, baseline_config: dict) -> None:
    stimulus = baseline_config['stimulus']
    arguments = dict(t_min=0, t_max=stimulus['duration_s'],
        gray_screen_dur=stimulus['gray_screen_s'],
        cpd=stimulus['spatial_frequency_cycles_per_degree'],
        temporal_f=stimulus['temporal_frequency_hz'],
        theta=stimulus['direction_deg'], contrast=stimulus['contrast_percent']/100.0)
    required = ('half_contrast_exact', 'combined_gray_exact', 'gray_is_zero',
                'retained_grating_exact', 'gray_join_frame_count_exact')
    required += ('zero_contrast_movie_all_zero',) if arguments['contrast'] == 0 else ('first_nonzero_frame_follows_gray',)
    if (movie.get('status') != 'SOURCE_MOVIE_CONTROL_WITNESS_PASS'
            or movie.get('effective_create_movie_arguments') != arguments
            or movie.get('gray_frames') != UNCHANGED_MOVIE_FRAMES
            or movie.get('first_retained_grating_local_frame') != 1
            or movie.get('source_default_phase') != 0.0
            or movie.get('frame_rate_hz') != 1000.0
            or movie.get('movie_shape') != [RATE_SAMPLES, 48, 96]
            or not all(movie.get('checks', {}).get(name) is True for name in required)):
        raise ValueError('SOURCE_CONTROL_MOVIE_CASE_OR_UNCHANGED_PREFIX_DIFFERS')


def validate_inventory(expected_ids, expected_models, actual_ids, actual_models) -> None:
    """Validate the ordered source inventory, including ID/model correspondence."""
    expected_ids, actual_ids = np.asarray(expected_ids), np.asarray(actual_ids)
    if (expected_ids.ndim != 1 or actual_ids.ndim != 1
            or expected_ids.dtype.kind not in 'iu' or actual_ids.dtype.kind not in 'iu'
            or len(np.unique(expected_ids)) != len(expected_ids)
            or len(np.unique(actual_ids)) != len(actual_ids)
            or len(expected_models) != len(expected_ids)
            or len(actual_models) != len(actual_ids)
            or not np.array_equal(expected_ids, actual_ids)
            or list(expected_models) != list(actual_models)):
        raise ValueError('SOURCE_CONTROL_REPRESENTATIVE_IDENTITY_DIFFERS')


def compare_rate_prefix(source_ids, baseline_clock, baseline_rates,
                        control_clock, control_rates, changed_frame: int) -> dict:
    """Compare exact prefixes; later changes are measured, never required per unit."""
    source_ids = np.asarray(source_ids)
    baseline_clock, control_clock = np.asarray(baseline_clock), np.asarray(control_clock)
    baseline_rates, control_rates = np.asarray(baseline_rates), np.asarray(control_rates)
    if (baseline_clock.ndim != 1 or control_clock.ndim != 1
            or not np.array_equal(baseline_clock, control_clock)):
        raise ValueError('SOURCE_CONTROL_CLOCK_DIFFERS')
    if (source_ids.ndim != 1 or source_ids.dtype.kind not in 'iu'
            or len(np.unique(source_ids)) != len(source_ids) or len(source_ids) == 0
            or not 0 < changed_frame < len(baseline_clock)
            or not np.isfinite(baseline_clock).all()
            or not (np.diff(baseline_clock) > 0).all()
            or baseline_rates.shape != (len(source_ids), len(baseline_clock))
            or control_rates.shape != baseline_rates.shape
            or not np.isfinite(baseline_rates).all() or not np.isfinite(control_rates).all()
            or (baseline_rates < 0).any() or (control_rates < 0).any()):
        raise ValueError('SOURCE_CONTROL_RATE_VALUES_OR_SHAPES_INVALID')
    differs = baseline_rates != control_rates
    delta = np.abs(control_rates - baseline_rates)
    prefix_counts = np.count_nonzero(differs[:, :changed_frame], axis=1)
    later_counts = np.count_nonzero(differs[:, changed_frame:], axis=1)
    prefix_maxima = delta[:, :changed_frame].max(axis=1)
    later_maxima = delta[:, changed_frame:].max(axis=1)
    return dict(clock_exact=True, source_units=len(source_ids), rate_units='Hz',
        unchanged_prefix=dict(frame_start=0, frame_stop_exclusive=changed_frame,
            first_time_s=float(baseline_clock[0]),
            last_time_s=float(baseline_clock[changed_frame-1]),
            exact=bool(not prefix_counts.any()), mismatch_count=int(prefix_counts.sum()),
            max_absolute_difference_hz=float(prefix_maxima.max()),
            affected_source_ids=source_ids[prefix_counts > 0].tolist()),
        at_and_after_changed_frame=dict(frame_start=changed_frame,
            frame_stop_exclusive=len(baseline_clock),
            first_time_s=float(baseline_clock[changed_frame]),
            mismatch_count=int(later_counts.sum()),
            max_absolute_difference_hz=float(later_maxima.max()),
            changed_source_ids=source_ids[later_counts > 0].tolist(),
            unchanged_source_ids=source_ids[later_counts == 0].tolist(),
            response_required_for_every_unit=False),
        per_unit=[dict(source_id=int(source_id), prefix_exact=bool(prefix_counts[i] == 0),
            prefix_mismatch_count=int(prefix_counts[i]),
            prefix_max_absolute_difference_hz=float(prefix_maxima[i]),
            later_changed=bool(later_counts[i] > 0), later_mismatch_count=int(later_counts[i]),
            later_max_absolute_difference_hz=float(later_maxima[i]))
            for i, source_id in enumerate(source_ids)])


@contextmanager
def observe_convolutions(signal_module, observations: list):
    """Observe the source's calls without changing arguments or forcing a method."""
    original = signal_module.convolve
    signature = inspect.signature(original)
    choose_method = signal_module.choose_conv_method

    @wraps(original)
    def observed(in1, in2, *args, **kwargs):
        bound = signature.bind(in1, in2, *args, **kwargs)
        bound.apply_defaults()
        mode, method = bound.arguments['mode'], bound.arguments['method']
        row = dict(call_index=len(observations),
            input_shapes=[list(np.shape(in1)), list(np.shape(in2))],
            input_dtypes=[str(np.asarray(in1).dtype), str(np.asarray(in2).dtype)],
            mode=mode, requested_method=method,
            selection_basis='choose_conv_method(measure=False)' if method == 'auto' else 'explicit',
            original_arguments_unchanged=True, completed=False)
        observations.append(row)
        try:
            row['chosen_method'] = (choose_method(in1, in2, mode=mode, measure=False)
                                    if method == 'auto' else method)
            result = original(in1, in2, *args, **kwargs)
            row['completed'] = True
            return result
        except BaseException as error:
            row.update(error_type=type(error).__name__, error=str(error))
            raise

    signal_module.convolve = observed
    try:
        yield
    finally:
        signal_module.convolve = original


def run_diagnostic(rate_archive: Path, movie_archive: Path, tables_archive: Path,
                   config_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = {}
    report = dict(status='FAILED_SOURCE_CONTROL_CAUSALITY_DIAGNOSTIC', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_rate_archive=str(rate_archive), source_movie_archive=str(movie_archive),
        source_tables_archive=str(tables_archive), source_runtime_qualified=False,
        source_fidelity_qualified=False, scientific_qualification=False,
        visual_action_mapping_qualified=False, original_visual_recipe_qualified=False,
        coupled_neural_episodes=0, production_episodes=0,
        tolerance_applied=False, convolution_calls=[],
        interpretation=('Exact unchanged-input prefix diagnostic only. A differing prefix is not '
                        'by itself biological noncausality; convolution method observations are '
                        'diagnostic evidence, not a causal attribution.'))
    try:
        config = yaml.safe_load(config_path.read_text())
        baseline, delivery = bound_json(rate_archive, 'probe/probe.json')
        movie, _ = bound_json(movie_archive, 'movie/report.json')
        if baseline.get('status') != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT':
            raise ValueError('SOURCE_CONTROL_EXECUTED_BASELINE_REQUIRED')
        validate_config_pair(baseline['config'], config)
        validate_movie_witness(movie, baseline['config'])
        cells_path = delivered_file(tables_archive, 'tables/lgn_full_col_cells_3.csv')
        models_path = delivered_file(tables_archive, 'tables/lgn_full_col_cell_models_3.csv')
        cells, models = read_source_cells(cells_path, models_path)
        if len(cells) != SOURCE_UNITS or len(models) != SOURCE_MODELS:
            raise ValueError('SOURCE_CONTROL_COMPLETE_SOURCE_DENOMINATOR_REQUIRED')
        validate_inventory(cells.index.to_numpy(), cells['model_id'].tolist(),
                           baseline['source_unit_ids'], baseline['source_models'])
        if (baseline.get('source_units') != SOURCE_UNITS
                or baseline.get('rate_shape') != [SOURCE_UNITS, RATE_SAMPLES]):
            raise ValueError('SOURCE_CONTROL_FULL_BASELINE_SHAPE_REQUIRED')
        representatives = cells.groupby('model_id', sort=False).head(1)
        expected_ids = [int(value) for value in representatives.index]
        expected_models = representatives['model_id'].tolist()
        if (len(representatives) != SOURCE_MODELS
                or set(expected_models) != set(models['model_id'])):
            raise ValueError('SOURCE_CONTROL_ALL_MODEL_REPRESENTATIVES_REQUIRED')
        report.update(config=config, config_sha256=sha256_file(config_path),
            baseline_config=baseline['config'], source_unit_ids=expected_ids,
            source_models=expected_models, changed_input_frame=UNCHANGED_MOVIE_FRAMES,
            baseline_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
            movie_report_sha256=sha256_file(movie_archive/'movie/report.json'),
            cells_sha256=sha256_file(cells_path), models_sha256=sha256_file(models_path))
        baseline_path = delivered_file(rate_archive, 'probe/firing_rates_f_tot.h5')
        report['baseline_rates_sha256'] = next(row['sha256'] for row in delivery['files']
            if row['path'] == 'probe/firing_rates_f_tot.h5')
        row_for_id = {source_id: row for row, source_id in enumerate(baseline['source_unit_ids'])}
        with h5py.File(baseline_path, 'r') as saved:
            if saved['firing_rates_Hz'].shape != (SOURCE_UNITS, RATE_SAMPLES):
                raise ValueError('SOURCE_CONTROL_BASELINE_DATA_SHAPE_DIFFERS')
            baseline_clock = saved['time'][:]
            baseline_rates = np.stack([saved['firing_rates_Hz'][row_for_id[source_id], :]
                                       for source_id in expected_ids])
        if baseline_clock.shape != (RATE_SAMPLES,):
            raise ValueError('SOURCE_CONTROL_BASELINE_CLOCK_SHAPE_DIFFERS')
        runtime = rate_archive/'prepared'
        load_prepared_runtime(runtime)
        cursor = importlib.import_module('lgnmodel.cursor')
        if Path(cursor.__file__).resolve() != (runtime/'lgnmodel/cursor.py').resolve():
            raise ValueError('SOURCE_CONTROL_CURSOR_ORIGIN_DIFFERS')
        report.update(prepared_runtime_sha256=sha256_file(runtime/'prepared.json'),
                      cursor_sha256=sha256_file(runtime/'lgnmodel/cursor.py'))
        with observe_convolutions(cursor.spsig, report['convolution_calls']):
            control = probe_runtime(runtime, tables_archive, config_path, output/'control')
        if not report['convolution_calls']:
            raise ValueError('SOURCE_CONTROL_CONVOLUTION_NOT_OBSERVED')
        if (control.get('status') != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
                or control.get('config') != config
                or control.get('config_sha256') != report['config_sha256']):
            raise ValueError('SOURCE_CONTROL_EXECUTED_CONFIG_DIFFERS')
        validate_inventory(expected_ids, expected_models,
                           control['source_unit_ids'], control['source_models'])
        with h5py.File(output/'control/firing_rates_f_tot.h5', 'r') as saved:
            comparison = compare_rate_prefix(expected_ids, baseline_clock, baseline_rates,
                saved['time'][:], saved['firing_rates_Hz'][:], UNCHANGED_MOVIE_FRAMES)
        prefix_exact = comparison['unchanged_prefix']['exact']
        report.update(comparison=comparison,
            status=('SOURCE_CONTROL_PREFIX_EXACT_DEVELOPMENT' if prefix_exact
                    else 'SOURCE_CONTROL_PREFIX_DIFFERS_DIAGNOSTIC'),
            exit_code=0 if prefix_exact else 2)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['convolution_chosen_method_counts'] = dict(Counter(
            row['chosen_method'] for row in report['convolution_calls'] if 'chosen_method' in row))
        report['runtime'] = runtime_manifest(config, list(sys.argv), report['exit_code'])
        report['artifacts'] = {relative: sha256_file(output/relative) for relative in
            ('control/probe.json', 'control/firing_rates_f_tot.h5') if (output/relative).is_file()}
        freeze_json(output/'report.json', report)
    return report


def validate_full_config_pair(baseline: dict, control: dict) -> None:
    """Compare consumed rate settings, excluding only unused spike metadata."""
    for config in (baseline, control):
        if (config.get('scope') != 'development_source_runtime_qualification'
                or config.get('unit_coverage') != 'all_source_units'):
            raise ValueError('SOURCE_CONTROL_BOTH_FULL_DEVELOPMENT_CONFIGS_REQUIRED')
    expected = {key: value for key, value in baseline.items() if key != 'spike_generation'}
    expected['stimulus'] = dict(baseline['stimulus'])
    expected['stimulus']['contrast_percent'] /= 2
    actual = {key: value for key, value in control.items() if key != 'spike_generation'}
    if actual != expected:
        raise ValueError('SOURCE_CONTROL_FULL_RATE_CONFIG_MAY_ONLY_HALVE_CONTRAST')


def validate_full_rate_receipt(reference: dict, delivery: dict) -> None:
    ids = np.asarray(reference.get('source_unit_ids', []))
    models = reference.get('source_models', [])
    if (delivery.get('exit_code') != 0
            or delivery.get('status') not in ('PASS', 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT')
            or reference.get('status') != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
            or reference.get('source_units') != SOURCE_UNITS
            or reference.get('complete_source_table_units') != SOURCE_UNITS
            or reference.get('rate_shape') != [SOURCE_UNITS, RATE_SAMPLES]
            or reference.get('shared_clock_checked_units') != SOURCE_UNITS
            or reference.get('clock_exactly_matches_source_separable_cursor') is not True
            or reference.get('time_units') != 's' or reference.get('firing_rate_units') != 'Hz'
            or ids.shape != (SOURCE_UNITS,) or ids.dtype.kind not in 'iu'
            or (ids < 0).any() or len(np.unique(ids)) != SOURCE_UNITS
            or len(models) != SOURCE_UNITS or len(set(models)) != SOURCE_MODELS):
        raise ValueError('SOURCE_CONTROL_COMPLETE_RATE_AND_EVERY_UNIT_CLOCK_RECEIPT_REQUIRED')


def compare_full_archives(rate_archive: Path, full_control_archive: Path,
                          movie_archive: Path, output: Path) -> dict:
    """Read complete archived rates; execute no movie, rate, spike, or native model."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_FULL_SOURCE_CONTROL_CAUSALITY_DIAGNOSTIC', exit_code=1,
        comparison_mode='full_population_archives',
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_rate_archive=str(rate_archive), full_control_archive=str(full_control_archive),
        source_movie_archive=str(movie_archive), source_runtime_qualified=False,
        source_fidelity_qualified=False, scientific_qualification=False,
        visual_action_mapping_qualified=False, original_visual_recipe_qualified=False,
        production_domain_qualified=False, coupled_neural_episodes=0, production_episodes=0,
        movie_recomputed=False, rates_recomputed=False, spikes_recomputed=False,
        tolerance_applied=False, rate_config_comparison_excluded_fields=['spike_generation'],
        rate_config_exclusion_reason='probe_runtime does not consume spike_generation metadata.',
        interpretation=('Exact full-population unchanged-input prefix comparison. A difference is '
                        'a diagnostic result, not by itself biological noncausality.'))
    try:
        baseline, baseline_delivery = bound_json(rate_archive, 'probe/probe.json')
        control, control_delivery = bound_json(full_control_archive, 'probe/probe.json')
        movie, _ = bound_json(movie_archive, 'movie/report.json')
        report.update(baseline_config=baseline['config'], config=control['config'],
            baseline_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
            control_report_sha256=sha256_file(full_control_archive/'probe/probe.json'),
            movie_report_sha256=sha256_file(movie_archive/'movie/report.json'))
        validate_full_rate_receipt(baseline, baseline_delivery)
        validate_full_rate_receipt(control, control_delivery)
        validate_full_config_pair(baseline['config'], control['config'])
        validate_movie_witness(movie, baseline['config'])
        validate_inventory(baseline['source_unit_ids'], baseline['source_models'],
                           control['source_unit_ids'], control['source_models'])
        report.update(source_unit_ids=baseline['source_unit_ids'], source_models=baseline['source_models'],
            source_units=SOURCE_UNITS, model_classes=SOURCE_MODELS,
            changed_input_frame=UNCHANGED_MOVIE_FRAMES, every_unit_clock_receipts_verified=True)
        manifests = {}
        for name, archive in (('baseline', rate_archive), ('control', full_control_archive)):
            manifests[name] = sha256_file(delivered_file(archive, 'prepared/prepared.json'))
        report['prepared_runtime_hashes'] = manifests
        if manifests['baseline'] != manifests['control']:
            raise ValueError('SOURCE_CONTROL_FULL_PREPARED_RUNTIME_DIFFERS')
        report['prepared_runtime_hashes_exact'] = True
        arrays = {}
        report['source_array_references'] = {}
        for name, archive, delivery in (('baseline', rate_archive, baseline_delivery),
                                         ('control', full_control_archive, control_delivery)):
            relative = 'probe/firing_rates_f_tot.h5'
            path = delivered_file(archive, relative)
            report['source_array_references'][name] = dict(archive=str(archive), path=relative,
                sha256=next(row['sha256'] for row in delivery['files'] if row['path'] == relative),
                time_dataset='time', rate_dataset='firing_rates_Hz',
                shape=[SOURCE_UNITS, RATE_SAMPLES], selected_rows='all_source_rows_in_declared_order')
            with h5py.File(path, 'r') as saved:
                if (saved['time'].shape != (RATE_SAMPLES,)
                        or saved['firing_rates_Hz'].shape != (SOURCE_UNITS, RATE_SAMPLES)):
                    raise ValueError('SOURCE_CONTROL_FULL_ARCHIVED_DATA_SHAPE_DIFFERS:' + name)
                clock = saved['time'][:]
                if not np.array_equal(clock, np.arange(RATE_SAMPLES)/1000.0):
                    raise ValueError('SOURCE_CONTROL_FULL_ARCHIVED_CURSOR_CLOCK_DIFFERS:' + name)
                arrays[name] = (clock, saved['firing_rates_Hz'][:])
        comparison = compare_rate_prefix(baseline['source_unit_ids'],
            arrays['baseline'][0], arrays['baseline'][1], arrays['control'][0], arrays['control'][1],
            UNCHANGED_MOVIE_FRAMES)
        exact = comparison['unchanged_prefix']['exact']
        report.update(comparison=comparison,
            status=('FULL_SOURCE_CONTROL_PREFIX_EXACT_DEVELOPMENT' if exact
                    else 'FULL_SOURCE_CONTROL_PREFIX_DIFFERS_DIAGNOSTIC'),
            exit_code=0 if exact else 2)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest(report.get('config', {}), list(sys.argv), report['exit_code'])
        report['artifacts'] = {}
        freeze_json(output/'report.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rate-archive', 'movie-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('tables-archive', 'config', 'full-control-archive'):
        parser.add_argument('--'+name, type=Path)
    arguments = parser.parse_args()
    if arguments.full_control_archive is not None:
        if arguments.tables_archive is not None or arguments.config is not None:
            parser.error('--full-control-archive uses archived configs; omit --tables-archive and --config')
        report = compare_full_archives(arguments.rate_archive, arguments.full_control_archive,
                                       arguments.movie_archive, arguments.output)
    else:
        if arguments.tables_archive is None or arguments.config is None:
            parser.error('representative execution requires --tables-archive and --config')
        report = run_diagnostic(arguments.rate_archive, arguments.movie_archive,
                                arguments.tables_archive, arguments.config, arguments.output)
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

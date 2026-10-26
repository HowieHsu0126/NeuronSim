"""Archive source-frame visual commands for one bound prospective V1 input case."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_configured_inputs import delivered_file
from .v1_circuit_stimulus import rate_config_without_noise
from .v1_lgn_control_causality import validate_movie_witness
from .v1_lgn_runtime import create_task_movie, load_prepared_runtime
from .v1_nest_input_replay import bound_json

SCHEMA = 'v1_source_visual_commands_v1'
LEGACY_FAMILY = 'legacy_generated_lgn'
CONSTRUCTED_FAMILY = 'constructed_lgn_background'
CONSTRUCTED_COMPLETE = 'CONSTRUCTED_VISUAL_INPUTS_COMPLETE_DEVELOPMENT'
SOURCE_UNITS = 17400
FRAME_COUNT = 3001
GRAY_FRAMES = 501
FRAME_RATE_HZ = 1000.0
NATIVE_DT_MS = 0.25
TICKS_PER_FRAME = 4
DURATION_TICKS = 12000
CONTROL_FEATURES = (
    {'name': 'grating_contrast_fraction', 'unit': 'dimensionless',
     'definition': 'Zero on source gray frames; source contrast_percent/100 on retained grating frames.'},
    {'name': 'spatial_frequency_cycles_per_source_degree', 'unit': 'cycles_per_source_visual_degree'},
    {'name': 'temporal_frequency_hz', 'unit': 'Hz'},
    {'name': 'source_theta_deg', 'unit': 'degree',
     'definition': 'Source thetaRad = pi*(180-theta)/180; not a cortical coordinate.'},
    {'name': 'source_phase_deg', 'unit': 'degree',
     'definition': 'Source phaseRad = pi*(180-phase)/180; source default phase is zero.'},
    {'name': 'grating_local_time_s', 'unit': 's',
     'definition': 'Retained source linspace samples; zero during the contrast-masked gray segment.'},
)


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_source_bindings(rate: dict, movie: dict, spikes: dict, *,
                             rate_report_sha256: str, rate_file_sha256: str,
                             prepared_runtime_sha256: str, spike_report_sha256: str,
                             _constructed_noise: bool = False) -> dict:
    """Bind existing successful inputs; do not infer a historical visual recipe."""
    ids = np.asarray(rate.get('source_unit_ids', []))
    if (rate.get('status') != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
            or rate.get('source_units') != SOURCE_UNITS
            or rate.get('rate_shape') != [SOURCE_UNITS, FRAME_COUNT]
            or ids.shape != (SOURCE_UNITS,) or ids.dtype.kind not in 'iu'
            or (ids < 0).any() or len(np.unique(ids)) != SOURCE_UNITS
            or rate.get('time_units') != 's' or rate.get('firing_rate_units') != 'Hz'
            or rate['config'].get('scope') != 'development_source_runtime_qualification'
            or rate['config'].get('unit_coverage') != 'all_source_units'):
        raise ValueError('V1_VISUAL_COMPLETE_SOURCE_RATE_CASE_REQUIRED')
    validate_movie_witness(movie, rate['config'])
    config = spikes['config']
    expected_spike_config = dict(rate['config'])
    # Older rate archives omit sampling metadata; explicit declarations must match.
    expected_spike_config.setdefault('spike_generation', config.get('spike_generation'))
    configs_match = (rate_config_without_noise(rate['config']) == rate_config_without_noise(config)
                     if _constructed_noise else config == expected_spike_config)
    trials = spikes.get('trials', [])
    if (spikes.get('status') != 'SOURCE_SPIKES_EXPORTED_EXACT'
            or spikes.get('source_units') != SOURCE_UNITS
            or spikes.get('rate_shape') != [SOURCE_UNITS, FRAME_COUNT]
            or not configs_match
            or (_constructed_noise and type(config['spike_generation'].get('trials')) is not int)
            or config.get('spike_generation', {}).get('trials') != 1
            or config['spike_generation']['starting_noise_seed'] != spikes.get('starting_noise_seed')
            or len(trials) != 1 or trials[0].get('trial') != 0
            or trials[0].get('source_units') != SOURCE_UNITS
            or trials[0].get('inventory_roundtrip_exact') is not True
            or trials[0].get('event_order_exact') is not True
            or trials[0].get('event_comparison', {}).get('all_event_pairs_exact') is not True
            or not _digest(trials[0].get('input_identity_sha256'))):
        raise ValueError('V1_VISUAL_COMPLETE_BOUND_SPIKE_CASE_REQUIRED')
    for key, expected in (('rate_report_sha256', rate_report_sha256),
                          ('rate_file_sha256', rate_file_sha256),
                          ('prepared_runtime_sha256', prepared_runtime_sha256)):
        if not _digest(expected) or spikes.get(key) != expected:
            raise ValueError('V1_VISUAL_SPIKE_SOURCE_BINDING_DIFFERS:' + key)
    if not _digest(spike_report_sha256):
        raise ValueError('V1_VISUAL_SPIKE_REPORT_IDENTITY_INVALID')
    return dict(config=config, stimulus=rate['config']['stimulus'], source_units=SOURCE_UNITS,
        rate_report_sha256=rate_report_sha256, rate_file_sha256=rate_file_sha256,
        prepared_runtime_sha256=prepared_runtime_sha256,
        movie_prepared_runtime_sha256=movie['prepared_runtime_sha256'],
        spike_report_sha256=spike_report_sha256,
        input_identity_sha256=trials[0]['input_identity_sha256'], trial=0)



def validate_constructed_source_bindings(rate: dict, movie: dict, lgn: dict,
                                         bundle: dict, **hashes) -> dict:
    """Join the complete LGN rate case to its separately verified LGN+BKG bundle."""
    bound = validate_source_bindings(rate, movie, lgn, _constructed_noise=True, **hashes)
    construction = rate.get('input_construction_binding')
    config = bound['config']
    expected_lgn = dict(lgn['trials'][0])
    expected_lgn.update(plan='lgn/'+expected_lgn['plan'], sonata='lgn/'+expected_lgn['sonata'])
    if (not isinstance(construction, dict) or not construction
            or lgn.get('input_construction_binding') != construction
            or bundle.get('input_construction_binding') != construction
            or bundle.get('status') != CONSTRUCTED_COMPLETE
            or bundle.get('complete_lgn_population') is not True
            or bundle.get('config') != config
            or bundle.get('source_clock') != dict(duration_ms=3000., resolution_ms=NATIVE_DT_MS)
            or bundle.get('rate_report_sha256') != bound['rate_report_sha256']
            or bundle.get('prepared_runtime_sha256') != bound['prepared_runtime_sha256']
            or bundle.get('lgn_report_sha256') != bound['spike_report_sha256']
            or bundle.get('artifacts', {}).get('lgn/report.json') != bound['spike_report_sha256']
            or not _digest(lgn.get('config_sha256'))
            or bundle.get('config_sha256') != lgn['config_sha256']
            or bundle.get('lgn_noise_seed') != config['spike_generation']['starting_noise_seed']
            or bundle.get('background_noise_seed') != config.get('background', {}).get('starting_noise_seed')
            or type(bundle.get('background_noise_seed')) is not int
            or bundle.get('populations', {}).get('lgn') != expected_lgn
            or bundle.get('populations', {}).get('bkg', {}).get('source_units') != 1
            or not _digest(bundle.get('populations', {}).get('bkg', {}).get('input_identity_sha256'))
            or not _digest(bundle.get('input_identity_sha256'))
            or not _digest(bundle.get('bundle_report_sha256'))
            or not _digest(movie.get('rate_report_sha256'))):
        raise ValueError('V1_VISUAL_CONSTRUCTED_SOURCE_BINDING_DIFFERS')
    bound.update(source_family=CONSTRUCTED_FAMILY, rate_config=rate['config'],
        lgn_report_sha256=bound.pop('spike_report_sha256'),
        lgn_input_identity_sha256=bound.pop('input_identity_sha256'),
        background_input_identity_sha256=bundle['populations']['bkg']['input_identity_sha256'],
        merged_input_identity_sha256=bundle['input_identity_sha256'],
        bundle_report_sha256=bundle['bundle_report_sha256'], bundle_status=bundle['status'],
        complete_lgn_population=True, input_construction_binding=construction,
        config_sha256=bundle['config_sha256'], source_clock=bundle['source_clock'],
        movie_rate_report_sha256=movie['rate_report_sha256'])
    return bound

def make_commands(stimulus: dict) -> dict:
    """Embed source frame indices without rounding any floating-point clock."""
    required = {'duration_s', 'gray_screen_s', 'spatial_frequency_cycles_per_degree',
                'temporal_frequency_hz', 'direction_deg', 'contrast_percent'}
    if (set(stimulus) != required
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) for value in stimulus.values())
            or stimulus['duration_s'] != 3.0 or stimulus['gray_screen_s'] != 0.5
            or not 0 <= stimulus['contrast_percent'] <= 100):
        raise ValueError('V1_VISUAL_BOUND_THREE_SECOND_SOURCE_RECIPE_REQUIRED')
    frames = np.arange(FRAME_COUNT, dtype=np.int64)
    local_clock = np.linspace(0, stimulus['duration_s']-stimulus['gray_screen_s'], 2501, endpoint=True)
    values = np.zeros((FRAME_COUNT, len(CONTROL_FEATURES)), dtype=np.float64)
    values[GRAY_FRAMES:, 0] = stimulus['contrast_percent']/100.0
    values[:, 1] = stimulus['spatial_frequency_cycles_per_degree']
    values[:, 2] = stimulus['temporal_frequency_hz']
    values[:, 3] = stimulus['direction_deg']
    values[:, 4] = 0.0
    values[GRAY_FRAMES:, 5] = local_clock[1:]
    return dict(frame_indices=frames, control_ticks=frames*TICKS_PER_FRAME,
                control_values=values, grating_source_clock_s=local_clock)


def reconstruct_frame(command, row_coordinates, col_coordinates):
    """Independent command rendering, preserving the source arithmetic order."""
    contrast, cpd, temporal_f, theta, phase, local_time = map(float, command)
    xx, yy = np.meshgrid(col_coordinates, row_coordinates, indexing='xy')
    if contrast == 0.0:
        return np.zeros_like(xx, dtype=np.float64)
    theta_rad = np.pi*(180-theta)/180.0
    phase_rad = np.pi*(180-phase)/180.0
    xy = xx*np.cos(theta_rad) + yy*np.sin(theta_rad)
    return contrast*np.sin(2*np.pi*(cpd*xy + temporal_f*local_time) + phase_rad)


def compare_source_movie(data, commands: dict) -> dict:
    values, rows, cols = (commands[key] for key in
                          ('control_values', 'row_coordinates', 'col_coordinates'))
    if (data.shape != (FRAME_COUNT, len(rows), len(cols))
            or values.shape != (FRAME_COUNT, len(CONTROL_FEATURES))
            or not np.isfinite(data).all() or not np.isfinite(values).all()):
        raise ValueError('V1_VISUAL_MOVIE_OR_COMMAND_VALUES_INVALID')
    counts = np.zeros(FRAME_COUNT, dtype=np.int64)
    maxima = np.zeros(FRAME_COUNT, dtype=np.float64)
    for index, command in enumerate(values):
        frame = reconstruct_frame(command, rows, cols)
        counts[index] = np.count_nonzero(data[index] != frame)
        maxima[index] = np.max(np.abs(data[index]-frame))
    commands['frame_mismatch_counts'] = counts
    commands['frame_max_absolute_differences'] = maxima
    return dict(all_values_exact=bool(not counts.any()), mismatch_count=int(counts.sum()),
        compared_values=int(data.size), max_absolute_difference=float(maxima.max()),
        affected_frame_indices=np.flatnonzero(counts).tolist(), tolerance_applied=False)


def validate_native_binding(report: dict, native: dict, *, native_family: str = 'legacy') -> None:
    if native_family not in ('legacy', 'pointnet'):
        raise ValueError('V1_VISUAL_NATIVE_REPORT_FAMILY_UNSUPPORTED')
    if native_family == 'legacy' and native.get('generated_lgn') is None:
        raise ValueError('V1_VISUAL_CONTROLS_REQUIRE_GENERATED_INPUT')
    if native_family == 'pointnet' and native.get('input_mode') != 'constructed_visual_inputs':
        raise ValueError('V1_VISUAL_CONTROLS_REQUIRE_CONSTRUCTED_INPUTS')
    if (report.get('schema') != SCHEMA
            or report.get('status') != 'SOURCE_VISUAL_COMMANDS_EXACT_DEVELOPMENT'
            or report.get('reconstruction', {}).get('all_values_exact') is not True
            or report['reconstruction'].get('mismatch_count') != 0
            or report.get('source_units') != SOURCE_UNITS
            or report.get('native_dt_ms') != NATIVE_DT_MS
            or report.get('duration_ticks') != DURATION_TICKS
            or report.get('control_features') != list(CONTROL_FEATURES)):
        raise ValueError('V1_VISUAL_EXACT_DEVELOPMENT_COMMANDS_REQUIRED')
    if native_family == 'legacy':
        if report.get('source_family', LEGACY_FAMILY) != LEGACY_FAMILY:
            raise ValueError('V1_VISUAL_NATIVE_REPORT_FAMILY_DIFFERS')
        generated = native['generated_lgn']
        for key in ('spike_report_sha256', 'input_identity_sha256', 'trial', 'config'):
            if key not in generated or report.get(key) != generated[key]:
                raise ValueError('V1_VISUAL_NATIVE_GENERATED_INPUT_DIFFERS:' + key)
        return
    if (report.get('source_family') != CONSTRUCTED_FAMILY
            or report.get('bundle_status') != CONSTRUCTED_COMPLETE
            or report.get('complete_lgn_population') is not True
            or report.get('trial') != 0
            or report.get('source_command_representation_verified') is not True
            or any(report.get(key) is not False for key in ('scientific_qualification',
                'source_fidelity_qualified', 'visual_action_mapping_qualified', 'production_domain_qualified'))
            or any(not _digest(report.get(key)) for key in ('lgn_report_sha256',
                'lgn_input_identity_sha256', 'background_input_identity_sha256', 'merged_input_identity_sha256'))):
        raise ValueError('V1_VISUAL_CONSTRUCTED_DEVELOPMENT_COMMANDS_REQUIRED')
    constructed = native.get('bindings', {}).get('spike_inputs', {})
    for report_key, native_key in (
            ('constructed_input_delivery_sha256', 'delivery_sha256'),
            ('constructed_input_files', 'files'), ('bundle_report_sha256', 'bundle_report_sha256'),
            ('merged_input_identity_sha256', 'input_identity_sha256'), ('bundle_status', 'bundle_status'),
            ('input_construction_binding', 'input_construction_binding'), ('config', 'config'),
            ('source_clock', 'source_clock'), ('complete_lgn_population', 'complete_lgn_population')):
        if report_key not in report or native_key not in constructed or report[report_key] != constructed[native_key]:
            raise ValueError('V1_VISUAL_NATIVE_CONSTRUCTED_INPUT_DIFFERS:' + report_key)


def load_visual_controls(archive: Path, native: dict, *, native_family: str = 'legacy') -> dict:
    """Opt in only for the native generated input already bound to this report."""
    report, _ = bound_json(archive, 'controls/report.json')
    validate_native_binding(report, native, native_family=native_family)
    path = delivered_file(archive, 'controls/commands.npz')
    if sha256_file(path) != report['artifacts']['commands.npz']:
        raise ValueError('V1_VISUAL_COMMAND_ARTIFACT_CHANGED')
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    expected = make_commands(report['config']['stimulus'])
    for key, value in expected.items():
        if (key not in arrays or arrays[key].dtype != value.dtype
                or not np.array_equal(arrays[key], value)):
            raise ValueError('V1_VISUAL_STORED_COMMAND_PROGRAM_DIFFERS:' + key)
    if (arrays.get('frame_mismatch_counts', np.ones(1)).shape != (FRAME_COUNT,)
            or np.any(arrays['frame_mismatch_counts'])
            or arrays.get('frame_max_absolute_differences', np.ones(1)).shape != (FRAME_COUNT,)
            or np.any(arrays['frame_max_absolute_differences'])):
        raise ValueError('V1_VISUAL_STORED_RECONSTRUCTION_DIFFERS')
    common = ('schema', 'trial', 'rate_report_sha256', 'rate_file_sha256', 'prepared_runtime_sha256',
        'movie_prepared_runtime_sha256', 'movie_report_sha256', 'source_movie_sha256',
        'config', 'native_dt_ms', 'duration_ticks', 'control_semantics')
    family_fields = ('spike_report_sha256', 'input_identity_sha256') if native_family == 'legacy' else (
        'lgn_report_sha256', 'lgn_input_identity_sha256', 'background_input_identity_sha256',
        'merged_input_identity_sha256', 'bundle_report_sha256', 'bundle_status',
        'constructed_input_delivery_sha256', 'constructed_input_files', 'input_construction_binding',
        'complete_lgn_population', 'config_sha256', 'source_clock', 'movie_rate_report_sha256',
        'movie_witness_artifact_sha256')
    record = {key: report[key] for key in (*common, *family_fields)}
    record.update(source_family=report.get('source_family', LEGACY_FAMILY),
                  control_report_sha256=sha256_file(archive/'controls/report.json'),
                  command_artifact_sha256=report['artifacts']['commands.npz'],
                  source_command_representation_verified=True, scientific_qualification=False,
                  source_fidelity_qualified=False, visual_action_mapping_qualified=False,
                  production_domain_qualified=False, original_visual_recipe_qualified=False)
    return dict(control_ticks=arrays['control_ticks'], control_values=arrays['control_values'],
                control_features=tuple(report['control_features']), control_kind='piecewise_constant',
                source_control_record=record)


def export_visual_controls(rate_archive: Path, movie_archive: Path,
                           spike_archive: Path, output: Path) -> dict:
    return _export_visual_controls(rate_archive, movie_archive, spike_archive, output)


def export_constructed_visual_controls(rate_archive: Path, movie_archive: Path,
                                      constructed_input_archive: Path, output: Path) -> dict:
    return _export_visual_controls(rate_archive, movie_archive, None, output,
                                   constructed_input_archive=constructed_input_archive)


def _export_visual_controls(rate_archive: Path, movie_archive: Path,
                            spike_archive: Path | None, output: Path, *,
                            constructed_input_archive: Path | None = None) -> dict:
    rate_archive, movie_archive, output = map(Path, (rate_archive, movie_archive, output))
    constructed = constructed_input_archive is not None
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status='FAILED_SOURCE_VISUAL_COMMAND_ARCHIVE', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        rate_archive=str(rate_archive), movie_archive=str(movie_archive),
        source_family=CONSTRUCTED_FAMILY if constructed else LEGACY_FAMILY,
        scientific_qualification=False, source_command_representation_verified=False,
        source_fidelity_qualified=False, visual_action_mapping_qualified=False,
        production_domain_qualified=False, original_visual_recipe_qualified=False,
        production_episodes=0, coupled_neural_episodes=0, lgn_rates_recomputed=False,
        spikes_recomputed=False, control_features=list(CONTROL_FEATURES),
        native_dt_ms=NATIVE_DT_MS, duration_ticks=DURATION_TICKS,
        control_semantics=('Piecewise constant source-frame commands: four native ticks per frame. '
            'The endpoint row is retained without defining an interval beyond the episode.'),
        spatial_coordinate_units='source_defined_visual_angle_degrees',
        cortical_coordinate_conversion_applied=False,
        temporal_origin='Source grating local linspace; first grating sample discarded by gray join.')
    try:
        rate, delivery = bound_json(rate_archive, 'probe/probe.json')
        movie_witness, _ = bound_json(movie_archive, 'movie/report.json')
        rate_path = delivered_file(rate_archive, 'probe/firing_rates_f_tot.h5')
        rate_digest = next(row['sha256'] for row in delivery['files']
                           if row['path'] == 'probe/firing_rates_f_tot.h5')
        runtime = rate_archive/'prepared'
        if constructed:
            # The queued no-controls consumer does not stage this producer-only module.
            from .v1_circuit_stimulus import load_inputs
            constructed_input_archive = Path(constructed_input_archive)
            _, bundle = load_inputs(constructed_input_archive,
                rate.get('input_construction_binding'), require_complete=True)
            spikes, bundle_delivery = bound_json(constructed_input_archive, 'visual_inputs/lgn/report.json')
            prepared = delivered_file(rate_archive, 'prepared/prepared.json')
            report.update(validate_constructed_source_bindings(rate, movie_witness, spikes, bundle,
                rate_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
                rate_file_sha256=rate_digest, prepared_runtime_sha256=sha256_file(prepared),
                spike_report_sha256=sha256_file(constructed_input_archive/'visual_inputs/lgn/report.json')))
            relatives = ('visual_inputs/report.json', *("visual_inputs/"+bundle['populations'][name][key]
                         for name in ('lgn', 'bkg') for key in ('plan', 'sonata')))
            # load_inputs has verified these exact files against delivery and producer hashes.
            files = {relative: {key: next(row for row in bundle_delivery['files'] if row['path']==relative)[key]
                                for key in ('bytes', 'sha256')} for relative in relatives}
            report.update(constructed_input_archive=str(constructed_input_archive),
                constructed_input_delivery_sha256=sha256_file(constructed_input_archive/'delivery.json'),
                constructed_input_files=files)
        else:
            spike_archive = Path(spike_archive)
            spikes, _ = bound_json(spike_archive, 'spikes/report.json')
            report['spike_archive'] = str(spike_archive)
            report.update(validate_source_bindings(rate, movie_witness, spikes,
                rate_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
                rate_file_sha256=rate_digest, prepared_runtime_sha256=sha256_file(runtime/'prepared.json'),
                spike_report_sha256=sha256_file(spike_archive/'spikes/report.json')))
        report['movie_report_sha256'] = sha256_file(movie_archive/'movie/report.json')
        functions, _ = load_prepared_runtime(runtime)
        report['source_movie_sha256'] = sha256_file(runtime/'lgnmodel/movie.py')
        movie = create_task_movie(functions, movie_witness['effective_create_movie_arguments'])
        if (list(movie.data.shape) != [FRAME_COUNT, 48, 96]
                or movie.frame_rate != FRAME_RATE_HZ):
            raise ValueError('V1_VISUAL_CURRENT_SOURCE_MOVIE_SHAPE_DIFFERS')
        commands = make_commands(report['stimulus'])
        with h5py.File(rate_path, 'r') as source:
            if source['firing_rates_Hz'].shape != (SOURCE_UNITS, FRAME_COUNT):
                raise ValueError('V1_VISUAL_RATE_DATA_SHAPE_DIFFERS')
            rate_clock = source['time'][:]
        if (not np.array_equal(rate_clock, np.arange(FRAME_COUNT)/FRAME_RATE_HZ)
                or not np.array_equal(movie.t_range, np.arange(FRAME_COUNT)*(1.0/FRAME_RATE_HZ))):
            raise ValueError('V1_VISUAL_SOURCE_CLOCK_EXPRESSION_DIFFERS')
        commands.update(row_coordinates=np.asarray(movie.row_range),
                        col_coordinates=np.asarray(movie.col_range),
                        movie_clock_s=np.asarray(movie.t_range), rate_clock_s=rate_clock)
        comparison = compare_source_movie(movie.data, commands)
        if constructed:
            witness_path = delivered_file(movie_archive, 'movie/movie_witness.npz')
            witness_digest = sha256_file(witness_path)
            if witness_digest != movie_witness['artifacts']['movie_witness.npz']:
                raise ValueError('V1_VISUAL_MOVIE_WITNESS_ARTIFACT_CHANGED')
            indices = np.array([0, GRAY_FRAMES-1, GRAY_FRAMES, FRAME_COUNT-1])
            expected = dict(frame_indices=indices, selected_frames=movie.data[indices],
                movie_clock_s=movie.t_range, rate_clock_s=rate_clock,
                row_coordinates=movie.row_range, col_coordinates=movie.col_range,
                frame_min=movie.data.min(axis=(1,2)), frame_max=movie.data.max(axis=(1,2)),
                origin_pixel=movie.data[:,0,0],
                center_pixel=movie.data[:,len(movie.row_range)//2,len(movie.col_range)//2])
            with np.load(witness_path, allow_pickle=False) as saved:
                if set(saved.files) != set(expected) or any(not np.array_equal(saved[key], value)
                                                          for key,value in expected.items()):
                    raise ValueError('V1_VISUAL_CURRENT_MOVIE_WITNESS_DIFFERS')
            report['movie_witness_artifact_sha256'] = witness_digest
        with (output/'commands.npz').open('xb') as stream:
            np.savez_compressed(stream, **commands)
        local_clock = commands['control_values'][GRAY_FRAMES:, 5]
        clock_differences = {}
        for name, clock in (('movie_absolute_minus_gray', movie.t_range),
                            ('rate_absolute_minus_gray', rate_clock)):
            candidate = clock[GRAY_FRAMES:]-report['stimulus']['gray_screen_s']
            clock_differences[name] = dict(mismatches=int(np.count_nonzero(candidate != local_clock)),
                max_absolute_difference_s=float(np.max(np.abs(candidate-local_clock))))
        exact = comparison['all_values_exact']
        report.update(status=('SOURCE_VISUAL_COMMANDS_EXACT_DEVELOPMENT' if exact
                              else 'SOURCE_VISUAL_COMMAND_RECONSTRUCTION_DIFFERS'),
            exit_code=0 if exact else 2, reconstruction=comparison,
            source_command_representation_verified=exact,
            gray_frames=GRAY_FRAMES, first_grating_frame=GRAY_FRAMES,
            first_grating_tick=GRAY_FRAMES*TICKS_PER_FRAME,
            first_grating_local_time_s=float(local_clock[0]),
            frame_count=FRAME_COUNT, movie_shape=list(movie.data.shape),
            source_movie_labels=list(movie.labels), source_movie_unit_labels=list(movie.units),
            movie_rate_clock_exact=bool(np.array_equal(movie.t_range, rate_clock)),
            movie_rate_clock_mismatches=int(np.count_nonzero(movie.t_range != rate_clock)),
            max_movie_rate_clock_difference_s=float(np.max(np.abs(movie.t_range-rate_clock))),
            local_clock_comparisons=clock_differences,
            native_frame_clock_exact=bool(np.array_equal(
                commands['control_ticks']*NATIVE_DT_MS/1000.0, rate_clock)),
            endpoint_control_retained=bool(commands['control_ticks'][-1] == DURATION_TICKS),
            prior_movie_manifest_equals_current=bool(
                report['movie_prepared_runtime_sha256'] == report['prepared_runtime_sha256']),
            prior_movie_manifest_interpretation=('Historical movie witness binds its own prepared manifest. '
                'This complete reconstruction executes the current rate-bound prepared source.'))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest(report.get('config', {}), list(sys.argv), report['exit_code'])
        report['artifacts'] = {'commands.npz': sha256_file(output/'commands.npz')} if (output/'commands.npz').is_file() else {}
        freeze_json(output/'report.json', report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rate-archive', 'movie-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--spike-archive', type=Path)
    source.add_argument('--constructed-input-archive', type=Path)
    args = parser.parse_args(argv)
    if args.constructed_input_archive is not None:
        return export_constructed_visual_controls(args.rate_archive, args.movie_archive,
            args.constructed_input_archive, args.output)['exit_code']
    return export_visual_controls(args.rate_archive, args.movie_archive,
                                  args.spike_archive, args.output)['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

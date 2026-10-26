"""Prepare the published LGN runtime and execute a declared development rate case."""
from __future__ import annotations

import argparse
import ast
import inspect
import difflib
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

from .provenance import freeze_json, sha256_file
from .v1_configured_inputs import delivered_file
from .v1_lgn_source import read_source_cells, read_input_construction_cells
from .v1_nest_input_replay import bound_json

SOURCE_FRAME_RATE_HZ = 1000.0


@contextmanager
def gray_movie_support(functions, contrast_percent):
    """Use the source grid/clock with zero values for the explicit gray task.

    The source grating constructor rejects contrast zero. This task extension
    builds its regular contrast-one movie and zeros its intensity before LGN
    filtering. Prepared source files and positive-contrast execution are intact.
    """
    if contrast_percent != 0:
        yield
        return
    original = functions.GratingMovie

    class GraySupportedGrating(original):
        def create_movie(self, *args, **kwargs):
            bound = inspect.signature(original.create_movie).bind(self, *args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            parameters.pop("self")
            gray = parameters["contrast"] == 0
            if gray:
                parameters["contrast"] = 1.0
            movie = super().create_movie(**parameters)
            if gray:
                movie.data.fill(0.0)
            return movie

    functions.GratingMovie = GraySupportedGrating
    try:
        yield
    finally:
        functions.GratingMovie = original


def create_task_movie(functions, arguments):
    """Render one task movie, including the defined zero-intensity condition."""
    with gray_movie_support(functions, arguments["contrast"] * 100):
        return functions.GratingMovie(120, 240).create_movie(**arguments)



def source_cross_from_above(values, threshold):
    """Preserve the executable crossing rule in Matplotlib 1.5.3 and 2.2.5."""
    values = np.asarray(values)
    indices = np.nonzero((values[:-1] >= threshold) & (values[1:] < threshold))[0]
    return indices + 1 if len(indices) else indices


def verify_crossing_reference(reference_archive, output):
    cases = [([], 0), ([0], 0), ([1, 0, -1, 0, -2, 2, 0], 0),
             ([-1, 0, 1], 0), ([2, 1, 0, 2, 1, -1], 1), ([0, 0, -1, -1], 0)]
    checks = []
    for tag in ('v1.5.3', 'v2.2.5'):
        source = delivered_file(reference_archive, tag+'.py')
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'cross_from_above')
        node.decorator_list = []
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        for values, threshold in cases:
            actual = source_cross_from_above(values, threshold)
            expected = namespace['cross_from_above'](values, threshold)
            exact = bool(np.array_equal(actual, expected) and actual.dtype == expected.dtype)
            checks.append(dict(version=tag, values=values, threshold=threshold, exact=exact,
                               reference_sha256=sha256_file(source)))
    report = dict(status='PASS' if all(c['exact'] for c in checks) else 'FAIL', comparisons=checks)
    freeze_json(output/'compatibility_validation.json', report)
    if report['status'] != 'PASS':
        raise ValueError('SOURCE_CROSSING_REFERENCE_MISMATCH')


def apply_source_compatibility(output, reference_archive):
    """Apply only observed library API repairs with source semantics preserved."""
    verify_crossing_reference(reference_archive, output)
    path = output/'lgnmodel/movie.py'
    before = path.read_text()
    original = ("        self.row_range  = np.linspace(0, self.row_size, self.row_size / physical_spacing, endpoint = True)\n"
                "        self.col_range  = np.linspace(0, self.col_size, self.col_size / physical_spacing, endpoint = True)\n")
    replacement = ("        row_samples = self.row_size / physical_spacing\n"
                   "        col_samples = self.col_size / physical_spacing\n"
                   "        if not (float(row_samples).is_integer() and float(col_samples).is_integer()):\n"
                   "            raise ValueError('SOURCE_SPATIAL_SAMPLE_COUNTS_MUST_BE_INTEGRAL')\n"
                   "        self.row_range = np.linspace(0, self.row_size, int(row_samples), endpoint=True)\n"
                   "        self.col_range = np.linspace(0, self.col_size, int(col_samples), endpoint=True)\n")
    if before.count(original) != 1:
        raise ValueError('SOURCE_GRID_COMPATIBILITY_CONTEXT_CHANGED')
    after = before.replace(original, replacement)
    path.write_text(after)
    (output/'compatibility.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile='language_converted/lgnmodel/movie.py', tofile='prepared/lgnmodel/movie.py')))
    path = output/'lgnmodel/util_fns.py'
    before = path.read_text()
    if before.count('mlab.cross_from_above(') != 1:
        raise ValueError('SOURCE_CROSSING_COMPATIBILITY_CONTEXT_CHANGED')
    after = before.replace('mlab.cross_from_above(', 'source_cross_from_above(')
    after += '\n\n' + inspect.getsource(source_cross_from_above)
    path.write_text(after)
    with (output/'compatibility.patch').open('a') as patch:
        patch.write(''.join(difflib.unified_diff(before.splitlines(keepends=True),
            after.splitlines(keepends=True), fromfile='language_converted/lgnmodel/util_fns.py',
            tofile='prepared/lgnmodel/util_fns.py')))
    path = output/'lgn_functions.py'
    before = path.read_text()
    node = next(n for n in ast.parse(before).body if isinstance(n, ast.FunctionDef)
                and n.name == 'calculate_firing_rate')
    function = ast.get_source_segment(before, node)
    assignment = '        firing_rates[counter + 1, :] = f_tot'
    if function.count(assignment) != 1:
        raise ValueError('SOURCE_RATE_CLOCK_CHECK_CONTEXT_CHANGED')
    checked = function.replace(assignment,
        "        if not np.array_equal(np.asarray(t), time):\n"
        "            raise ValueError('SOURCE_UNIT_RATE_CLOCK_DIFFERS:' + str(node[0]))\n" + assignment)
    after = before.replace(function, checked, 1)
    path.write_text(after)
    with (output/'compatibility.patch').open('a') as patch:
        patch.write(''.join(difflib.unified_diff(before.splitlines(keepends=True),
            after.splitlines(keepends=True), fromfile='language_converted/lgn_functions.py',
            tofile='prepared/lgn_functions.py')))
    return ['Explicit integer NumPy spatial sample counts; nonintegral counts are not admitted.',
            'Restore removed Matplotlib crossing function; exact executable-rule checks against versions 1.5.3 and 2.2.5.',
            'Require each unit to return the same time grid before recording its rates.']


def prepare_runtime(members_archive, legacy_archive, reference_archive, output, converter_python):
    intake, _ = bound_json(members_archive, 'runtime/intake.json')
    if intake['status'] != 'SOURCE_RUNTIME_MEMBERS_ACQUIRED':
        raise ValueError('SOURCE_RUNTIME_COMPLETE_MEMBERS_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    originals = {}
    for row in intake['members']:
        name = row['path']
        source = delivered_file(members_archive, 'runtime/lgnmodel/' + name)
        target = output/'lgnmodel'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[str(target.relative_to(output))] = sha256_file(source)
    source = delivered_file(legacy_archive, 'lgn_functions.py/source_asset.txt')
    shutil.copy2(source, output/'lgn_functions.py')
    originals['lgn_functions.py'] = sha256_file(source)
    command = [converter_python, '-m', 'lib2to3', '-w', '-n',
               str(output/'lgnmodel'), str(output/'lgn_functions.py')]
    with (output/'conversion.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    compatibility = apply_source_compatibility(output, reference_archive) if result.returncode == 0 else []
    record = dict(compatibility_edits=compatibility, status='SOURCE_RUNTIME_PREPARED' if result.returncode == 0 else 'SOURCE_CONVERSION_FAILED',
                  command=command, exit_code=result.returncode,
                  original_archive_complete=False, source_runtime_qualified=False,
                  source_members_archive=str(members_archive), legacy_source_archive=str(legacy_archive),
                  files=[dict(path=name, original_sha256=originals[name],
                         prepared_sha256=sha256_file(output/name)) for name in originals])
    freeze_json(output/'prepared.json', record)
    if result.returncode:
        raise RuntimeError('SOURCE_LANGUAGE_CONVERSION_FAILED')
    return record



def load_prepared_runtime(runtime):
    runtime = runtime.resolve()
    prepared = json.loads((runtime/'prepared.json').read_text())
    if prepared['status'] != 'SOURCE_RUNTIME_PREPARED':
        raise ValueError('SOURCE_RUNTIME_PREPARATION_REQUIRED')
    for row in prepared['files']:
        if sha256_file(runtime/row['path']) != row['prepared_sha256']:
            raise ValueError('SOURCE_PREPARED_MODULE_CHANGED:' + row['path'])
    previous = sys.modules.get('lgn_functions')
    if previous is not None and Path(previous.__file__).resolve().parent != runtime:
        raise RuntimeError('SOURCE_RUNTIME_REQUIRES_ITS_OWN_PROCESS')
    sys.path.insert(0, str(runtime))
    functions = importlib.import_module('lgn_functions')
    poisson = importlib.import_module('lgnmodel.poissongeneration')
    if (Path(functions.__file__).resolve() != runtime/'lgn_functions.py'
            or Path(poisson.__file__).resolve() != runtime/'lgnmodel/poissongeneration.py'):
        raise RuntimeError('SOURCE_RUNTIME_MODULE_ORIGIN_DIFFERS')
    return functions, poisson


class SourceNodes:
    """The read-only node iteration interface used by the published rate function."""
    def __init__(self, cells):
        self.records = []
        for source_id, row in cells.iterrows():
            values = row.to_dict()
            values['position'] = (values['x'], values['y'])
            self.records.append((int(source_id), values))

    def nodes(self):
        return self.records


def probe_runtime(runtime, tables_archive, config_path, output, reference_rate_archive=None,
                  *, input_construction_archive=None):
    if (tables_archive is None) == (input_construction_archive is None):
        raise ValueError('SOURCE_EXACTLY_ONE_CELL_INPUT_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text())
    if config['scope'] != 'development_source_runtime_qualification':
        raise ValueError('SOURCE_RUNTIME_DEVELOPMENT_SCOPE_REQUIRED')
    report = dict(status='FAILED_SOURCE_RATE_PROBE', config=config,
                  config_sha256=sha256_file(config_path), source_runtime_qualified=False,
                  source_fidelity_qualified=False, original_visual_recipe_qualified=False,
                  coupled_neural_episodes=0, production_episodes=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        if input_construction_archive is None:
            cells, models = read_source_cells(
                delivered_file(tables_archive, 'tables/lgn_full_col_cells_3.csv'),
                delivered_file(tables_archive, 'tables/lgn_full_col_cell_models_3.csv'))
        else:
            cells, models, binding = read_input_construction_cells(input_construction_archive)
            report['input_construction_binding'] = binding
        if len(cells) != 17400 or len(models) != 14:
            raise ValueError('SOURCE_FULL_LGN_DENOMINATOR_REQUIRED')
        report['complete_source_table_units'] = len(cells)
        reference = None
        if reference_rate_archive is not None:
            reference, _ = bound_json(reference_rate_archive, 'probe/probe.json')
            if (reference['status'] != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
                    or reference['config']['stimulus'] != config['stimulus']):
                raise ValueError('SOURCE_RATE_REFERENCE_CASE_DIFFERS')
            if reference.get('input_construction_binding') != report.get('input_construction_binding'):
                raise ValueError('SOURCE_RATE_REFERENCE_INPUT_CONSTRUCTION_DIFFERS')
        if config['unit_coverage'] == 'model_representatives':
            cells = cells.groupby('model_id', sort=False).head(1)
        elif config['unit_coverage'] != 'all_source_units':
            raise ValueError('SOURCE_UNIT_COVERAGE_INVALID')
        report.update(source_unit_ids=[int(i) for i in cells.index],
                      source_models=cells['model_id'].tolist(), source_units=len(cells))
        functions, _ = load_prepared_runtime(runtime)
        stimulus = config['stimulus']
        prefix = output/'firing_rates'
        start = time.perf_counter()
        with gray_movie_support(functions, stimulus['contrast_percent']):
            functions.calculate_firing_rate(SourceNodes(cells), 'gratings', str(prefix),
                stimulus['duration_s'], stimulus['gray_screen_s'],
                stimulus['spatial_frequency_cycles_per_degree'], stimulus['temporal_frequency_hz'],
                stimulus['direction_deg'], stimulus['contrast_percent'])
        if stimulus['contrast_percent'] == 0:
            report['task_movie_extension'] = 'source_grid_and_clock_with_all_intensities_zero_before_LGN_filtering'
        report['rate_execution_seconds'] = time.perf_counter() - start
        with h5py.File(str(prefix)+'_f_tot.h5', 'r') as source:
            clock = source['time'][:]
            rates = source['firing_rates_Hz'][:]
        if (clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all()
                or not (np.diff(clock) > 0).all() or rates.shape != (len(cells), len(clock))
                or not np.isfinite(rates).all() or (rates < 0).any()):
            raise ValueError('SOURCE_RATE_CLOCK_OR_VALUES_INVALID')
        cursor_clock = np.arange(len(clock)) / SOURCE_FRAME_RATE_HZ
        if not np.array_equal(clock, cursor_clock):
            raise ValueError('SOURCE_EVALUATOR_CLOCK_MISMATCH')
        constructor_clock = np.arange(len(clock)) * (1.0/SOURCE_FRAME_RATE_HZ)
        legacy_clock = np.linspace(0, stimulus['duration_s'], len(clock))
        if reference is not None:
            path = delivered_file(reference_rate_archive, 'probe/firing_rates_f_tot.h5')
            source_rows = {int(source_id): i for i, source_id in enumerate(cells.index)}
            rows = [source_rows[int(source_id)] for source_id in reference['source_unit_ids']]
            with h5py.File(path, 'r') as previous:
                clock_equal = bool(np.array_equal(clock, previous['time'][:]))
                rates_equal = bool(np.array_equal(rates[rows], previous['firing_rates_Hz'][:]))
            report['prior_representative_comparison'] = dict(source_archive=str(reference_rate_archive),
                source_units=len(rows), clock_exact=clock_equal, rates_exact=rates_equal)
            if not clock_equal or not rates_equal:
                raise ValueError('SOURCE_RATE_REPRESENTATIVE_READBACK_DIFFERS')
        rate_clock_guard = rates.max(axis=1) * np.max(np.diff(clock*1000.0)) / 1000.0
        legacy_clock_guard = rates.max(axis=1) * np.max(np.diff(legacy_clock*1000.0)) / 1000.0
        report.update(shared_clock_checked_units=len(cells),
            clock_exactly_matches_source_separable_cursor=True,
            clock_mismatches_vs_movie_constructor=int(np.count_nonzero(clock != constructor_clock)),
            max_clock_difference_vs_movie_constructor_s=float(np.max(np.abs(clock-constructor_clock))),
            units_exceeding_source_bernoulli_guard_on_rate_clock=int((rate_clock_guard > 0.1).sum()),
            units_exceeding_source_bernoulli_guard_on_legacy_clock=int((legacy_clock_guard > 0.1).sum()))
        report.update(status='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT', time_units='s',
            firing_rate_units='Hz', rate_shape=list(rates.shape),
            first_time_s=float(clock[0]), last_time_s=float(clock[-1]),
            clock_exactly_matches_source_movie_constructor=bool(np.array_equal(clock, constructor_clock)),
            clock_exactly_matches_legacy_wrapper_linspace=bool(np.array_equal(clock, legacy_clock)),
            min_rate_hz=float(rates.min()), max_rate_hz=float(rates.max()),
            per_unit_max_rate_hz=rates.max(axis=1).tolist())
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output/'probe.json', report)
    return report


def inspect_stimulus(rate_archive, output):
    """Execute source movie controls without repeating any LGN rate computation."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_MOVIE_WITNESS',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_rate_archive=str(rate_archive), lgn_rates_recomputed=False,
                  original_visual_recipe_qualified=False, visual_action_mapping_qualified=False,
                  spatial_unit_annotation_qualified=False, production_episodes=0)
    try:
        reference, _ = bound_json(rate_archive, 'probe/probe.json')
        if reference['status'] != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT':
            raise ValueError('SOURCE_MOVIE_REQUIRES_EXECUTED_RATE_RECIPE')
        functions, _ = load_prepared_runtime(rate_archive/'prepared')
        stimulus = reference['config']['stimulus']
        arguments = dict(t_min=0, t_max=stimulus['duration_s'],
            gray_screen_dur=stimulus['gray_screen_s'], cpd=stimulus['spatial_frequency_cycles_per_degree'],
            temporal_f=stimulus['temporal_frequency_hz'], theta=stimulus['direction_deg'],
            contrast=stimulus['contrast_percent']/100.0)
        movie = create_task_movie(functions, arguments)
        with h5py.File(delivered_file(rate_archive, 'probe/firing_rates_f_tot.h5'), 'r') as saved:
            rate_clock = saved['time'][:]
        checks = dict(finite_movie=bool(np.isfinite(movie.data).all()),
                      movie_and_rate_frame_count_equal=len(movie.t_range) == len(rate_clock),
                      gray_present=arguments['gray_screen_dur'] > 0)
        if not all(checks.values()):
            raise ValueError('SOURCE_MOVIE_BASELINE_OR_CLOCK_INVALID')
        gray = functions.FullFieldFlashMovie(movie.row_range, movie.col_range,
            arguments['gray_screen_dur']+1, arguments['gray_screen_dur']+2,
            frame_rate=movie.frame_rate).full(t_max=arguments['gray_screen_dur'])
        grating = create_task_movie(functions, dict(arguments,
            gray_screen_dur=0, t_max=arguments['t_max']-arguments['gray_screen_dur']))
        gray_frames = len(gray.t_range)
        checks.update(gray_is_zero=bool(np.all(gray.data == 0)),
                      combined_gray_exact=bool(np.array_equal(movie.data[:gray_frames], gray.data)),
                      retained_grating_exact=bool(np.array_equal(movie.data[gray_frames:], grating.data[1:])),
                      gray_join_frame_count_exact=len(movie.t_range) == gray_frames+len(grating.t_range)-1)
        del gray, grating
        half = create_task_movie(functions, dict(arguments, contrast=arguments['contrast']/2))
        checks['half_contrast_exact'] = bool(np.array_equal(half.data, movie.data/2))
        del half
        stationary = create_task_movie(functions, dict(arguments, temporal_f=0))
        checks['zero_tf_movie_stationary'] = bool(np.all(
            stationary.data[gray_frames:] == stationary.data[gray_frames]))
        del stationary
        active_frames = np.flatnonzero(np.any(movie.data != 0, axis=(1, 2)))
        if arguments['contrast'] == 0:
            checks['zero_contrast_movie_all_zero'] = bool(not len(active_frames))
        else:
            checks['baseline_movie_temporally_varying'] = bool(np.any(movie.data[gray_frames:] != movie.data[gray_frames]))
            checks['first_nonzero_frame_follows_gray'] = bool(len(active_frames) and active_frames[0] == gray_frames)
        indices = np.array([0, gray_frames-1, gray_frames, len(movie.t_range)-1])
        np.savez_compressed(output/'movie_witness.npz', frame_indices=indices,
            selected_frames=movie.data[indices], movie_clock_s=movie.t_range, rate_clock_s=rate_clock,
            row_coordinates=movie.row_range, col_coordinates=movie.col_range,
            frame_min=movie.data.min(axis=(1, 2)), frame_max=movie.data.max(axis=(1, 2)),
            origin_pixel=movie.data[:, 0, 0], center_pixel=movie.data[:, len(movie.row_range)//2, len(movie.col_range)//2])
        report.update(status='SOURCE_MOVIE_CONTROL_WITNESS_PASS' if all(checks.values()) else 'SOURCE_MOVIE_CONTROL_WITNESS_DIFFERS',
            checks=checks, effective_create_movie_arguments=arguments,
            source_default_phase=0.0, movie_shape=list(movie.data.shape), frame_rate_hz=movie.frame_rate,
            source_labels=list(movie.labels), source_unit_labels=list(movie.units),
            source_spatial_size_comment='GratingMovie row_size and col_size are commented in degrees; returned Movie labels remain pixel.',
            row_coordinate_endpoints=[float(movie.row_range[0]), float(movie.row_range[-1])],
            col_coordinate_endpoints=[float(movie.col_range[0]), float(movie.col_range[-1])],
            row_first_spacing=float(movie.row_range[1]-movie.row_range[0]),
            col_first_spacing=float(movie.col_range[1]-movie.col_range[0]),
            gray_frames=gray_frames, last_gray_movie_time_s=float(movie.t_range[gray_frames-1]),
            first_grating_movie_time_s=float(movie.t_range[gray_frames]),
            first_retained_grating_local_frame=1, first_retained_grating_local_time_s=1.0/movie.frame_rate,
            movie_rate_clock_exact=bool(np.array_equal(movie.t_range, rate_clock)),
            movie_rate_clock_mismatches=int(np.count_nonzero(movie.t_range != rate_clock)),
            max_movie_rate_clock_difference_s=float(np.max(np.abs(movie.t_range-rate_clock))),
            diagnostic_controls_only={'contrast_fraction': arguments['contrast']/2, 'temporal_frequency_hz': 0.0},
            full_movie_archived=False, retained_frame_indices=indices.tolist(),
            rate_report_sha256=sha256_file(rate_archive/'probe/probe.json'),
            prepared_runtime_sha256=sha256_file(rate_archive/'prepared/prepared.json'))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['artifacts'] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        freeze_json(output/'report.json', report)
    return report


def inspect_spatial_units(rate_archive, movie_archive, tables_archive, output):
    """Check source Gaussian coordinate consumption on the existing movie grid."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_SPATIAL_COORDINATE_CHECK',
        started_utc=datetime.now(timezone.utc).isoformat(), source_fidelity_qualified=False,
        production_episodes=0, independent_biological_calibration=False,
        source_spatial_labels_modified=False, lgn_rates_recomputed=False,
        numeric_coordinate_unit='source_defined_visual_angle_degrees',
        unit_basis='Source GratingMovie degree comments and cpd definition; source spatial_size and Gaussian spacing ratios.',
        scope='Base Gaussian primitives for every source model representative; no axis-orientation or whole-movie rescaling claim.',
        cases=[])
    arrays = {}
    try:
        reference, _ = bound_json(rate_archive, 'probe/probe.json')
        movie, _ = bound_json(movie_archive, 'movie/report.json')
        if (reference['status'] != 'SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
                or movie['status'] != 'SOURCE_MOVIE_CONTROL_WITNESS_PASS'
                or movie['rate_report_sha256'] != sha256_file(rate_archive/'probe/probe.json')):
            raise ValueError('SOURCE_SPATIAL_RATE_MOVIE_BINDING_DIFFERS')
        cells, models = read_source_cells(delivered_file(tables_archive, 'tables/lgn_full_col_cells_3.csv'),
            delivered_file(tables_archive, 'tables/lgn_full_col_cell_models_3.csv'))
        selected = cells.loc[reference['source_unit_ids']]
        if len(selected) != len(models) or set(selected['model_id']) != set(models['model_id']):
            raise ValueError('SOURCE_SPATIAL_COMPLETE_MODEL_REPRESENTATIVES_REQUIRED')
        path = delivered_file(movie_archive, 'movie/movie_witness.npz')
        if sha256_file(path) != movie['artifacts']['movie_witness.npz']:
            raise ValueError('SOURCE_SPATIAL_OBSERVED_GRID_CHANGED')
        with np.load(path, allow_pickle=False) as witness:
            row, col = witness['row_coordinates'], witness['col_coordinates']
        arrays.update(row_coordinates=row, col_coordinates=col)
        functions, _ = load_prepared_runtime(rate_archive/'prepared')
        for source_id, cell in selected.iterrows():
            sigma = (float(cell['spatial_size']/3.0),)*2
            translate = (float(cell['x']), float(cell['y']))
            baseline = functions.GaussianSpatialFilter(translate=translate, sigma=sigma,
                                                       origin=(0., 0.)).get_kernel(row, col)
            scaled = functions.GaussianSpatialFilter(translate=tuple(2*x for x in translate),
                sigma=tuple(2*x for x in sigma), origin=(0., 0.)).get_kernel(row*2, col*2)
            finite = bool(len(baseline.kernel) and np.isfinite(baseline.kernel).all()
                          and np.isfinite(scaled.kernel).all())
            exact = {name: bool(np.array_equal(getattr(baseline, name), getattr(scaled, name)))
                     for name in ('row_inds', 'col_inds', 'kernel')}
            report['cases'].append(dict(source_id=int(source_id), model_id=cell['model_id'],
                translation=list(translate), sigma=list(sigma), coordinate_scale=2.0,
                finite_nonempty=finite, comparisons=exact))
            for name in ('row_inds', 'col_inds', 'kernel'):
                arrays[f'unit_{int(source_id)}_baseline_{name}'] = getattr(baseline, name)
                arrays[f'unit_{int(source_id)}_scaled_{name}'] = getattr(scaled, name)
        np.savez_compressed(output/'spatial_kernels.npz', **arrays)
        passed = all(row['finite_nonempty'] and all(row['comparisons'].values()) for row in report['cases'])
        report.update(status='SOURCE_SPATIAL_COORDINATE_CONSUMPTION_EXACT' if passed else 'SOURCE_SPATIAL_COORDINATE_CONSUMPTION_DIFFERS',
            source_units=len(selected), model_types=len(models), source_unit_labels=movie['source_unit_labels'],
            sigma_basis='spatial_size / 3 defines Gaussian standard deviation in the same angular unit.',
            gaussian_source_sha256=sha256_file(rate_archive/'prepared/lgnmodel/spatialfilter.py'),
            movie_source_sha256=sha256_file(rate_archive/'prepared/lgnmodel/movie.py'),
            rate_source_sha256=sha256_file(rate_archive/'prepared/lgn_functions.py'),
            source_grid_sha256=sha256_file(path))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['artifacts'] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
        freeze_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('members-archive', 'legacy-archive', 'reference-archive', 'output'):
        prepare.add_argument('--'+name, type=Path, required=True)
    prepare.add_argument('--converter-python', required=True)
    probe = commands.add_parser('probe')
    for name in ('runtime', 'config', 'output'):
        probe.add_argument('--'+name, type=Path, required=True)
    source = probe.add_mutually_exclusive_group(required=True)
    source.add_argument('--tables-archive', type=Path)
    source.add_argument('--input-construction', type=Path)
    probe.add_argument('--reference-rate-archive', type=Path)
    movie = commands.add_parser('inspect-stimulus')
    movie.add_argument('--rate-archive', type=Path, required=True)
    movie.add_argument('--output', type=Path, required=True)
    spatial = commands.add_parser('inspect-spatial-units')
    for name in ('rate-archive', 'movie-archive', 'tables-archive', 'output'):
        spatial.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        prepare_runtime(args.members_archive, args.legacy_archive, args.reference_archive, args.output, args.converter_python)
    elif args.mode == 'probe':
        probe_runtime(args.runtime, args.tables_archive, args.config, args.output, args.reference_rate_archive,
                      input_construction_archive=args.input_construction)
    elif args.mode == 'inspect-spatial-units':
        result = inspect_spatial_units(args.rate_archive, args.movie_archive, args.tables_archive, args.output)
        print(json.dumps(result, indent=2))
        if result['status'] != 'SOURCE_SPATIAL_COORDINATE_CONSUMPTION_EXACT':
            raise SystemExit(2)
    else:
        result = inspect_stimulus(args.rate_archive, args.output)
        print(json.dumps(result, indent=2))
        if result['status'] != 'SOURCE_MOVIE_CONTROL_WITNESS_PASS':
            raise SystemExit(2)


if __name__ == '__main__':
    main()

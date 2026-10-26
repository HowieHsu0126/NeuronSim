"""Independently replay source host currents against a reconciled native trace.

Build the adjacent C++ source with g++ -std=c++11 -O3. The engine accepts only
the population map and known ordered commands. Its seed, reset-draw prefix,
distribution parameters and schedule are fixed from source, never fitted to
the trace. Exact replay is scoped to the recorded compiler/standard library.
This verifies external currents, not independent stochastic streams or recall.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING, Iterable

import numpy as np

from . import ca3_native_observation as observation
from .ca3_control_trace import BACKGROUND_SIGMA, source_commands
from .ca3_retrieval_readouts import DURATION_MS, PYRAMIDAL, source_identity
from .provenance import sha256_file
if TYPE_CHECKING:
    from .ca3_realization import SourceRealization

SOURCE_ORDER = tuple(BACKGROUND_SIGMA)
SOURCE_FILE = Path(__file__).with_suffix('.cpp')


def engine_input(groups: Iterable[observation.Group], commands: dict) -> bytes:
    groups = tuple(groups)
    by_name = {g.name: g for g in groups}
    ordered = sorted(groups, key=lambda g: g.start)
    if (len(groups) != 8 or set(by_name) != set(SOURCE_ORDER)
            or ordered[0].start != 0
            or any(g.units <= 0 for g in groups)
            or any(a.stop != b.start for a, b in zip(ordered, ordered[1:]))):
        raise ValueError('CA3_REPLAY_COMPLETE_SOURCE_POPULATIONS_REQUIRED')
    times, units = np.asarray(commands['time_ms']), np.asarray(commands['unit_ids'])
    if (times.ndim != 1 or units.shape != times.shape
            or times.dtype.kind not in 'iu' or units.dtype.kind not in 'iu'):
        raise ValueError('CA3_REPLAY_ORDERED_INTEGER_COMMANDS_REQUIRED')
    rows = [f'{by_name[name].start} {by_name[name].units}' for name in SOURCE_ORDER]
    rows.append(str(len(times)))
    rows.extend(f'{int(t)} {int(u)}' for t, u in zip(times, units))
    return ('\n'.join(rows)+'\n').encode('ascii')


def compare_currents(engine: Path, groups, commands, intervals, *,
                     realization: SourceRealization | None = None, expected_stop_ms=DURATION_MS) -> dict:
    """Exhaust both streams; count every unequal float bit, including late errors."""
    groups = tuple(groups)
    if type(expected_stop_ms) is not int or expected_stop_ms not in (DURATION_MS, 1210):
        raise ValueError('CA3_REPLAY_DURATION_UNSUPPORTED')
    payload = engine_input(groups, commands)
    units = sum(g.units for g in groups)
    compared = mismatches = steps = 0
    first = []
    by_group = {g.name: 0 for g in groups}
    maximum_error = 0.0
    with tempfile.TemporaryFile(mode='w+b') as errors:
        arguments = [] if realization is None else realization.replay_arguments()
        if expected_stop_ms != DURATION_MS:
            arguments += ['--duration-ms', str(expected_stop_ms)]
        if realization is not None and realization.phase != 'retrieval':
            raise ValueError('CA3_CURRENT_REPLAY_REQUIRES_RETRIEVAL_REALIZATION')
        process = subprocess.Popen([str(Path(engine).resolve()), *arguments], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            process.stdin.write(payload)
            process.stdin.close()
            for interval in intervals:
                if (interval.start_ms != steps or interval.stop_ms != steps+1
                        or steps >= expected_stop_ms):
                    raise ValueError('CA3_REPLAY_COMPLETE_ORDERED_CLOCK_REQUIRED')
                actual = interval.effective_current
                if actual.shape != (units,) or actual.dtype != np.dtype('<f4') or not np.isfinite(actual).all():
                    raise ValueError('CA3_REPLAY_FULL_FINITE_FLOAT32_CURRENT_REQUIRED')
                raw = process.stdout.read(units*4)
                if len(raw) != units*4:
                    raise ValueError('CA3_REPLAY_ENGINE_OUTPUT_TRUNCATED')
                expected = np.frombuffer(raw, dtype='<f4')
                if not np.isfinite(expected).all():
                    raise ValueError('CA3_REPLAY_ENGINE_NONFINITE')
                different = actual.view('<u4') != expected.view('<u4')
                n = int(np.count_nonzero(different))
                if n:
                    for group in groups:
                        by_group[group.name] += int(np.count_nonzero(different[group.start:group.stop]))
                    maximum_error = max(maximum_error, float(np.max(np.abs(
                        actual.astype(np.float64)-expected.astype(np.float64)))))
                    for unit in np.flatnonzero(different)[:max(0, 8-len(first))]:
                        first.append(dict(time_ms=steps, unit_id=int(unit),
                            actual=float(actual[unit]), expected=float(expected[unit]),
                            actual_bits=int(actual.view('<u4')[unit]),
                            expected_bits=int(expected.view('<u4')[unit])))
                mismatches += n
                compared += units
                steps += 1
            if steps != expected_stop_ms:
                raise ValueError('CA3_REPLAY_COMPLETE_ORDERED_CLOCK_REQUIRED')
            if process.stdout.read(1):
                raise ValueError('CA3_REPLAY_ENGINE_OUTPUT_TRAILING')
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError('CA3_REPLAY_ENGINE_FAILED: '+errors.read(4096).decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
    return dict(intervals_compared=steps, current_values_compared=compared,
                mismatched_values=mismatches, mismatches_by_population=by_group,
                first_mismatches=first, maximum_absolute_difference_pa=maximum_error,
                exact_float32_bits_equal=mismatches == 0, tolerance=0)


def bound_inputs(observed: Path, control_report: Path, *, coupled_extension: bool, source_realization_config=None):
    identity = source_identity(coupled_extension=coupled_extension)
    control = json.loads(control_report.read_text())
    native = json.loads((observed/'native_report.json').read_text())
    if source_realization_config is not None or 'source_realization' in native:
        from .ca3_realization import bind_runtime

        if not coupled_extension:
            raise ValueError('CA3_REALIZATION_COUPLED_HOST_REPLAY_REQUIRED')
        bind_runtime(observed, native, source_realization_config)
        identity['overlay_sha256'] = native['overlay_source_sha256']
        if (control.get('source_realization') != native['source_realization']
                or control.get('realization_circuit_binding') != native.get('realization_circuit_binding')):
            raise ValueError('CA3_REALIZATION_CONTROL_NATIVE_IDENTITY_DIFFERS')
    required = dict(status=('VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY' if coupled_extension
                            else 'VERIFIED_SOURCE_CA3_CONTROL_DELIVERY'),
                    units=identity['core_units'], intervals_verified=DURATION_MS,
                    actual_current_values_checked=DURATION_MS*identity['core_units'],
                    numeric_conversion_applied=False)
    if any(control.get(k) != v for k, v in required.items()):
        raise ValueError('CA3_REPLAY_VERIFIED_CONTROL_DELIVERY_REQUIRED')
    if control.get('current_units', {}).get('model_specific_physical_unit') != 'pA':
        raise ValueError('CA3_REPLAY_PINNED_PA_CURRENT_REQUIRED')
    if control.get('observer_implementation_sha256') != sha256_file(Path(observation.__file__)):
        raise ValueError('CA3_REPLAY_OBSERVATION_READER_CHANGED')
    bindings = control['input_sha256']
    if bindings['main_source'] != identity['overlay_sha256'] or bindings['config_source'] != identity['config_sha256']:
        raise ValueError('CA3_REPLAY_PINNED_SOURCE_REQUIRED')
    hashes = {name: sha256_file(observed/name) for name in
              ('native_report.json', 'native_observation.bin', 'trainPat.csv')}
    if any(hashes[name] != bindings[name] for name in hashes) or hashes['trainPat.csv'] != identity['pattern_sha256']:
        raise ValueError('CA3_REPLAY_CONTROL_INPUT_CHANGED')
    if source_realization_config is not None:
        hashes['realization_source.cpp'] = sha256_file(observed/'realization_source.cpp')
        hashes['source_realization.yaml'] = sha256_file(observed/'source_realization.yaml')
        if hashes['source_realization.yaml'] != sha256_file(source_realization_config):
            raise ValueError('CA3_REALIZATION_PACKAGED_CONFIGURATION_DIFFERS')
    trace = observed/'native_observation.bin'
    if (native['native_observation_sha256'] != hashes['native_observation.bin']
            or native['native_observation_bytes'] != trace.stat().st_size):
        raise ValueError('CA3_REPLAY_NATIVE_TRACE_CHANGED')
    units, groups = observation.read_header(trace)
    if units != identity['core_units'] or {g.name: g.units for g in groups} != identity['populations']:
        raise ValueError('CA3_REPLAY_PINNED_UNIT_IDENTITIES_REQUIRED')
    pattern = np.asarray([int(v) for v in (observed/'trainPat.csv').read_text().split()], dtype=np.int64)
    pyr = next(g for g in groups if g.name == PYRAMIDAL)
    commands = source_commands(pattern, pyr.start, coupled_extension=coupled_extension)
    if len(commands['unit_ids']) != control['ordered_assignments_verified']:
        raise ValueError('CA3_REPLAY_SOURCE_COMMAND_COUNT_CHANGED')
    return identity, groups, commands, hashes


def analyze(observed, control_report, engine, output, *, coupled_extension=False,
            input_origins=None, source_realization_config=None) -> dict:
    observed, control_report, engine, output = map(Path, (observed, control_report, engine, output))
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    identity, groups, commands, hashes = bound_inputs(observed, control_report,
        coupled_extension=coupled_extension, source_realization_config=source_realization_config)
    realization = None
    if source_realization_config is not None:
        from .ca3_realization import load_realization

        realization = load_realization(source_realization_config)
    arguments = [] if realization is None else realization.replay_arguments()
    engine_identity = json.loads(subprocess.check_output([str(engine.resolve()), *arguments, '--identity'], text=True))
    if realization is not None:
        expected = dict(explicit_noise_block=True, uniform_seed=realization.uniform_seed,
                        common_engine_seed=realization.engine_seed, post_seed_discard_draws_per_unit=0)
        if any(engine_identity.get(k) != v for k,v in expected.items()):
            raise ValueError('CA3_REALIZATION_HOST_ENGINE_IDENTITY_DIFFERS')
    circuit_seed = 10 if realization is None else realization.circuit_seed
    if engine_identity.get('carlsim_seed') != circuit_seed or engine_identity.get('reset_draws_per_unit') != 9:
        raise ValueError('CA3_REPLAY_FIXED_SOURCE_INITIALIZATION_REQUIRED')
    checks = compare_currents(engine, groups, commands, observation.iter_intervals(
        observed/'native_observation.bin', expected_stop_ms=DURATION_MS), realization=realization)
    passed = checks['exact_float32_bits_equal']
    units = identity['core_units']
    pyr_units = identity['populations'][PYRAMIDAL]
    report = dict(status='VERIFIED_CA3_HOST_CURRENT_REPLAY' if passed else 'CA3_HOST_CURRENT_REPLAY_DIFFERENT',
        created_utc=datetime.now(timezone.utc).isoformat(), **checks, units=units,
        coupled_extension=coupled_extension, source_executions=1,
        input_origins=input_origins or dict(observed=str(observed), control_report=str(control_report)),
        input_sha256=dict(hashes, control_report=sha256_file(control_report)),
        engine_identity=engine_identity, engine_sha256=sha256_file(engine),
        implementation_sha256=sha256_file(Path(__file__)), cpp_source_sha256=sha256_file(SOURCE_FILE),
        rng_contract=dict(carlsim_srand48_seed=circuit_seed, pre_current_drand48_draws=9*units,
            background_intervals=941, recall_intervals=60,
            external_current_drand48_draws=941*units+60*pyr_units,
            default_cpp_engines=9, persistent_distribution_objects=9,
            source_draw_order=list(SOURCE_ORDER), fitted_state_or_parameters=False),
        current_unit='pA', numeric_conversion_applied=False,
        source_host_current_rng_reconstructed=passed, independent_rng_streams_qualified=False,
        future_realized_current_model_visible=False, native_simulation_executed=False,
        pattern_completion_qualified=False, scientific_fidelity_qualified=False,
        source_scale_qualified=False, production_episodes=0,
        elapsed_seconds=time.monotonic()-started)
    if realization is not None:
        from .ca3_realization import contract

        report['source_realization'] = contract(realization)
        report['rng_contract'].update(pre_current_drand48_draws=0,
            explicit_post_setup_uniform_seed=realization.uniform_seed,
            common_cpp_engine_seed=realization.engine_seed,
            source_topology_noise_coupling_preserved=False)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observed', 'control-report', 'engine', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--observed-origin', type=Path)
    parser.add_argument('--control-report-origin', type=Path)
    parser.add_argument('--coupled-extension', action='store_true')
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    report = analyze(args.observed, args.control_report, args.engine, args.output,
                     coupled_extension=args.coupled_extension, source_realization_config=args.source_realization_config,
                     input_origins=dict(observed=str(args.observed_origin or args.observed),
                                        control_report=str(args.control_report_origin or args.control_report)))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['exact_float32_bits_equal'] else 1)


if __name__ == '__main__':
    main()

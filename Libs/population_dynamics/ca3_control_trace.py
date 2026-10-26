"""Bind source-declared CA3 cue commands to the reconciled native current trace.

The exported command events are known inputs. Realized background currents are
source provenance, not future model inputs. This checks delivery of the pinned
source protocol; it neither reconstructs its RNG draws nor tests causal recall.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import ca3_native_observation as observation
from .ca3_retrieval_readouts import (
    ASSEMBLY_STARTS, ASSEMBLY_UNITS, CONFIG_SHA256, CORE_UNITS, DURATION_MS,
    ELIGIBLE_UNITS, OVERLAY_SHA256, PATTERN_SHA256, POPULATIONS, PYRAMIDAL, source_identity, _source_proof,
)
from .provenance import sha256_file
from .carlsim_records import validate_coupled_identity

CUE_STARTS_MS = (520, 750, 980)
CUE_DURATION_MS = 20
SOURCE_SET_CURRENT = np.float32(10000)
BACKGROUND_SIGMA = {
    PYRAMIDAL: 1.5, 'CA3_Axo_Axonic': 1.25, 'CA3_Basket': 1.75,
    'CA3_BC_CCK': 1., 'CA3_Bistratified': 1., 'CA3_Ivy': 1.75,
    'CA3_MFA_ORDEN': 1.25, 'CA3_QuadD_LM': 1.25,
}


def source_commands(pattern, pyramidal_start, *, coupled_extension=False):
    """Keep the ordered source assignments, including repeated IDs in a phase."""
    identity = source_identity(coupled_extension=coupled_extension)
    pattern = np.asarray(pattern)
    if (pattern.shape != (identity['pattern_integer_count'],) or pattern.dtype.kind not in 'iu'
            or np.any(pattern < 0) or np.any(pattern >= identity['assembly_units'])):
        raise ValueError('CA3_SOURCE_PATTERN_INDICES_INVALID')
    if type(pyramidal_start) is not int or pyramidal_start < 0:
        raise ValueError('CA3_PYRAMIDAL_START_INVALID')
    times, ids, cues = [], [], []
    per_phase = len(pattern) // CUE_DURATION_MS
    for cue, (start, offset) in enumerate(zip(CUE_STARTS_MS, identity['assembly_starts']), 1):
        for phase in range(CUE_DURATION_MS):
            selected = pattern[phase*per_phase:(phase+1)*per_phase]
            selected = selected[selected < identity['eligible_units']]
            times.extend([start+phase]*len(selected))
            ids.extend((selected.astype(np.int64)+offset+pyramidal_start).tolist())
            cues.extend([cue]*len(selected))
    return dict(time_ms=np.asarray(times, dtype=np.int32),
                unit_ids=np.asarray(ids, dtype=np.int64),
                cue_id=np.asarray(cues, dtype=np.int8),
                set_current=np.full(len(ids), SOURCE_SET_CURRENT, dtype=np.float32))


def source_phase(time_ms):
    for cue, start in enumerate(CUE_STARTS_MS, 1):
        if start <= time_ms < start+CUE_DURATION_MS:
            return f'cue_{cue}', cue
    if time_ms < CUE_STARTS_MS[0]:
        return 'initial_background', 0
    for cue, start in enumerate(CUE_STARTS_MS[:-1], 1):
        if start+CUE_DURATION_MS <= time_ms < CUE_STARTS_MS[cue]:
            return f'background_after_cue_{cue}', 0
    return 'background_after_cue_3', 0


def check_currents(groups, intervals, commands, *, expected_stop_ms=DURATION_MS):
    """Stream the complete source schedule; aggregate actual raw float currents."""
    groups = tuple(groups)
    if type(expected_stop_ms) is not int or expected_stop_ms not in (DURATION_MS, 1210):
        raise ValueError('CA3_CONTROL_DURATION_UNSUPPORTED')
    units = max(g.stop for g in groups)
    if {g.name for g in groups} != set(BACKGROUND_SIGMA):
        raise ValueError('CA3_SOURCE_POPULATIONS_REQUIRED')
    if sorted(i for g in groups for i in range(g.start, g.stop)) != list(range(units)):
        raise ValueError('CA3_SOURCE_UNIT_MAP_INVALID')
    expected = {t: commands['unit_ids'][commands['time_ms'] == t]
                for start in CUE_STARTS_MS for t in range(start, start+CUE_DURATION_MS)}
    statistics, held = {}, {}
    assigned, held_values, steps = 0, 0, 0
    previous = None
    for interval in intervals:
        t = interval.start_ms
        if t != steps or interval.stop_ms != t+1 or t >= expected_stop_ms:
            raise ValueError('CA3_COMPLETE_ORDERED_NATIVE_CLOCK_REQUIRED')
        current = interval.effective_current
        if current.shape != (units,) or current.dtype != np.dtype('<f4') or not np.isfinite(current).all():
            raise ValueError('CA3_FULL_FINITE_FLOAT32_CURRENT_REQUIRED')
        if np.any(current < 0):
            raise ValueError('CA3_SOURCE_CURRENT_NEGATIVE')
        phase, cue = source_phase(t)
        if cue:
            indices = expected[t]
            if not np.all(current[indices].view('<u4') == SOURCE_SET_CURRENT.view('<u4')):
                raise ValueError('CA3_DECLARED_CUE_CURRENT_NOT_DELIVERED')
            assigned += len(indices)
            if t in CUE_STARTS_MS:
                if previous is None:
                    raise ValueError('CA3_PRE_CUE_CURRENT_REQUIRED')
                held = {g.name: previous[g.start:g.stop].copy() for g in groups if g.name != PYRAMIDAL}
        elif np.any(current > SOURCE_SET_CURRENT):
            raise ValueError('CA3_SOURCE_BACKGROUND_CAP_VIOLATION')
        for group in groups:
            values = current[group.start:group.stop]
            if cue and group.name != PYRAMIDAL:
                if not np.array_equal(values.view('<u4'), held[group.name].view('<u4')):
                    raise ValueError('CA3_INHIBITORY_CURRENT_HOLD_MISMATCH')
                held_values += len(values)
            key = (phase, group.name)
            row = statistics.setdefault(key, dict(phase=phase, group=group.name,
                start_ms=t, stop_ms=t+1, units=group.units, current_values=0,
                current_sum=0., current_squared_sum=0., minimum_raw=float('inf'),
                maximum_raw=0., exact_source_set_value_count=0))
            row['stop_ms'] = t+1
            row['current_values'] += len(values)
            row['current_sum'] += float(np.sum(values, dtype=np.float64))
            row['current_squared_sum'] += float(np.sum(values.astype(np.float64)**2))
            row['minimum_raw'] = min(row['minimum_raw'], float(values.min()))
            row['maximum_raw'] = max(row['maximum_raw'], float(values.max()))
            row['exact_source_set_value_count'] += int(np.count_nonzero(values == SOURCE_SET_CURRENT))
        previous = current.copy()
        steps += 1
    if steps != expected_stop_ms or assigned != len(commands['unit_ids']):
        raise ValueError('CA3_COMPLETE_SOURCE_SCHEDULE_REQUIRED')
    rows = []
    for row in statistics.values():
        row['mean_raw'] = row['current_sum']/row['current_values']
        row['second_moment_raw'] = row['current_squared_sum']/row['current_values']
        rows.append(row)
    return rows, dict(intervals_verified=steps, ordered_assignments_verified=assigned,
                      inhibitory_held_unit_ms_verified=held_values,
                      actual_current_values_checked=steps*units)


def background_schedule(groups):
    rows = []
    boundaries = (0, 520, 540, 750, 770, 980, 1000, 1001)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        phase, cue = source_phase(start)
        for group in groups:
            is_pyramidal = group.name == PYRAMIDAL
            held = bool(cue and not is_pyramidal)
            rows.append(dict(phase=phase, start_ms=start, stop_ms=stop, group=group.name,
                update='hold_previous_realization' if held else 'refresh_each_ms',
                held_from_ms=start-1 if held else None,
                cpp_lognormal_m=None if held else 4.,
                cpp_lognormal_s=None if held else 1. if cue else BACKGROUND_SIGMA[group.name],
                multiplier=None if held else 'float32(drand48())',
                background_cap_raw=None if held or (cue and is_pyramidal) else float(SOURCE_SET_CURRENT),
                model_visible='distribution_and_command_schedule',
                realized_future_current_model_visible=False))
    return rows


def _write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(observed, reconciliation, main_source, config_source, output, *, coupled_extension=False,
            source_realization_config=None):
    identity = source_identity(coupled_extension=coupled_extension)
    observed, reconciliation, main_source, config_source, output = map(
        Path, (observed, reconciliation, main_source, config_source, output))
    if output.exists():
        raise FileExistsError(output)
    proof = json.loads(reconciliation.read_text())
    native = json.loads((observed/'native_report.json').read_text())
    if source_realization_config is not None or 'source_realization' in native:
        from .ca3_realization import bind_program

        if not coupled_extension:
            raise ValueError('CA3_REALIZATION_COUPLED_CONTROL_REQUIRED')
        bind_program(native, source_realization_config, main_source)
        identity['overlay_sha256'] = native['overlay_source_sha256']
        if (proof.get('source_realization') != native['source_realization']
                or proof.get('realization_circuit_binding') != native.get('realization_circuit_binding')):
            raise ValueError('CA3_REALIZATION_CONTROL_RECONCILIATION_IDENTITY_DIFFERS')
    if coupled_extension:
        validate_coupled_identity(native)
        _source_proof(native,coupled_extension=True, source_realization_config=source_realization_config, runtime=observed)
        if any(proof.get(k) != native[k] for k in ('coupled_extension','coupled_extension_config')):
            raise ValueError('CA3_CONTROL_COUPLED_RECONCILIATION_IDENTITY_DIFFERS')
    required = dict(status=('PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY' if coupled_extension
                            else 'PASS_NATIVE_CA3_OBSERVATION_REPLAY'), units=identity['core_units'],
        duration_ms=DURATION_MS, actual_current_trace_verified=True,
        native_internal_spike_counter_verified=True, event_multisets_equal=True,
        terminal_native_clock_independently_verified=True,
        observation_source_execution_equivalence_qualified=True)
    if any(proof.get(k) != v for k, v in required.items()):
        raise ValueError('CA3_COMPLETED_NATIVE_OBSERVATION_RECONCILIATION_REQUIRED')
    if (proof['helper_implementation_sha256']['ca3_native_observation']
            != sha256_file(Path(observation.__file__))):
        raise ValueError('CA3_RECONCILED_OBSERVATION_READER_CHANGED')
    trace = observed/'native_observation.bin'
    hashes = {name: sha256_file(path) for name, path in {
        'native_report.json': observed/'native_report.json', 'native_observation.bin': trace,
        'trainPat.csv': observed/'trainPat.csv', 'main_source': main_source,
        'config_source': config_source, 'reconciliation': reconciliation}.items()}
    if (hashes['native_report.json'] != proof['observed_report_sha256']
            or hashes['native_observation.bin'] != proof['native_observation_sha256']
            or hashes['native_observation.bin'] != native['native_observation_sha256']
            or trace.stat().st_size != native['native_observation_bytes']
            or hashes['trainPat.csv'] != native['pattern_sha256'] or hashes['trainPat.csv'] != identity['pattern_sha256']
            or hashes['main_source'] != identity['overlay_sha256'] or hashes['config_source'] != identity['config_sha256']):
        raise ValueError('CA3_CONTROL_TRACE_SOURCE_OR_INPUT_CHANGED')
    units, groups = observation.read_header(trace)
    if units != identity['core_units'] or {g.name: g.units for g in groups} != identity['populations']:
        raise ValueError('CA3_PINNED_SOURCE_UNIT_IDENTITIES_REQUIRED')
    pattern = np.asarray([int(v) for v in (observed/'trainPat.csv').read_text().split()], dtype=np.int64)
    pyr = next(g for g in groups if g.name == PYRAMIDAL)
    commands = source_commands(pattern, pyr.start,coupled_extension=coupled_extension)
    rows, checks = check_currents(groups, observation.iter_intervals(trace, expected_stop_ms=DURATION_MS), commands)
    output.mkdir(parents=True)
    np.savez(output/'cue_command_events.npz', **commands)
    _write_csv(output/'background_schedule.csv', background_schedule(groups))
    _write_csv(output/'actual_current_summary.csv', rows)
    report = dict(status=('VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY' if coupled_extension
                         else 'VERIFIED_SOURCE_CA3_CONTROL_DELIVERY'),
        created_utc=datetime.now(timezone.utc).isoformat(), **checks, units=units,
        source_executions=1, source_cues=3, input_sha256=hashes,
        observation_map=dict(time_unit='ms', interval_ms=1,
            command_time='start of the native current-update interval',
            unit_ids='native global neuron IDs',
            groups=[dict(group_id=g.group_id, name=g.name, start=g.start, stop_exclusive=g.stop)
                    for g in groups]),
        implementation_sha256=sha256_file(Path(__file__)),
        observer_implementation_sha256=sha256_file(Path(observation.__file__)),
        command_semantics='ordered source assignments setting selected cells to10000raw; not additive current',
        background_semantics='source C++ lognormal constructor parameters; uniform multiplier; no moment reinterpretation',
        actual_currents_role='source_provenance_not_future_model_input',
        never_boosted_membership='source-command eligibility, not inferred from observed current magnitude',
        # The observer preserves the generic API's mA comment. This pinned
        # biological model instead defines its current in pA (Kopsick 2024,
        # Table 3). Its nS*mV synaptic current and raw I/C voltage update use
        # that same scale; changing stored numbers would introduce an error.
        observer_current_units=proof['current_units'],
        current_units=dict(proof['current_units'], model_specific_physical_unit='pA'),
        current_unit_basis=dict(model_doi='10.1007/s10827-024-00881-3',
            model_location='Table 3 and Par50',
            intrinsic_model_doi='10.1371/journal.pcbi.1007462',
            intrinsic_model_location='Equation 1 and mV/ms trace calibration',
            numeric_interpretation='raw pA; nS*mV synaptic current; C in pF gives mV/ms',
            numeric_current_response_job=580725),
        numeric_conversion_applied=False,
        source_rng_draws_independently_reconstructed=False, independent_rng_streams_qualified=False,
        causal_claim_qualified=False, pattern_completion_qualified=False,
        scientific_fidelity_qualified=False, full_scale_qualified=False, production_episodes=0,
        inferential_statistics_performed=False,
        artifacts={name:sha256_file(output/name) for name in
            ('cue_command_events.npz', 'background_schedule.csv', 'actual_current_summary.csv')})
    if coupled_extension:
        report.update(coupled_extension=native['coupled_extension'],
                      coupled_extension_config=native['coupled_extension_config'])
    if source_realization_config is not None:
        report['source_realization'] = native['source_realization']
        report['realization_circuit_binding'] = native['realization_circuit_binding']
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observed', 'reconciliation', 'main-source', 'config-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--coupled-extension',action='store_true')
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.observed, args.reconciliation, args.main_source, args.config_source, args.output,coupled_extension=args.coupled_extension, source_realization_config=args.source_realization_config), indent=2))


if __name__ == '__main__':
    main()

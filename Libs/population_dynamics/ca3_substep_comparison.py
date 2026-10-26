"""Matched-input retrieval sensitivity to five, ten, twenty and optional forty neuron substeps.

This is a descriptive experiment on one fixed learned graph and configured
initialization. No tolerance is fitted, no convergence order is assumed, and
no formation or biological-fidelity qualification is produced.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import zip_longest
import json
from pathlib import Path
import time

import numpy as np

from .ca3_cue_contrast import (INPUT_FIELDS, LIBRARIES, PHASE_EDGES_MS, PHASE_NAMES,
                              bind_run, bind_source_evidence, field_counts)
from . import ca3_native_observation as observer
from .ca3_native_observation_replay import _events, _compare_counters
from .ca3_retrieval_readouts import DURATION_MS
from .provenance import sha256_file


def bind_runs(reference, ten, twenty, source_analysis, source_rng, *, forty=None):
    runs = {5: bind_run(reference, omission=False),
            10: bind_run(ten, omission=False, neuron_substeps=10),
            20: bind_run(twenty, omission=False, neuron_substeps=20)}
    if forty is not None:
        runs[40] = bind_run(forty, omission=False, neuron_substeps=40)
    baseline = runs[5]
    _, proofs = bind_source_evidence(baseline, source_analysis, source_rng)
    for count in tuple(runs)[1:]:
        run = runs[count]
        if run.groups != baseline.groups:
            raise ValueError('CA3_REFINEMENT_UNIT_MAP_DIFFERS')
        for key in (*INPUT_FIELDS, 'coupled_extension', 'native_observation_overlay',
                    'observer_implementation_sha256', 'conductance_memory_overlay',
                    'conductance_memory_implementation_sha256'):
            if run.native[key] != baseline.native[key]:
                raise ValueError('CA3_REFINEMENT_INITIALIZATION_OR_BACKEND_DIFFERS:' + key)
        for key in ('resolved', 'sha256'):
            if run.native['coupled_extension_config'][key] != baseline.native['coupled_extension_config'][key]:
                raise ValueError('CA3_REFINEMENT_CONFIGURATION_DIFFERS')
        paired = json.loads((run.root/'manifest.json').read_text())['paired_backend']
        libraries = {row['path']: row['sha256'] for row in paired['libraries']}
        if (paired.get('backend_library_bytes_equal') is not True
                or paired['manifest_sha256'] != baseline.hashes['manifest.json']
                or paired['source_build_sha256'] != baseline.hashes['build_report.json']
                or set(libraries) != set(LIBRARIES)
                or any(run.hashes[name] != baseline.hashes[name] or libraries[name] != run.hashes[name]
                       for name in LIBRARIES)):
            raise ValueError('CA3_REFINEMENT_PAIRED_BACKEND_BYTES_DIFFER')
    return runs, proofs


def compare_inputs(traces, units):
    """Compare every requested and effective float32 input; preserve sign bits."""
    readers = [observer.iter_intervals(path, expected_stop_ms=DURATION_MS) for path in traces]
    intervals = 0
    for values in zip_longest(*readers):
        if any(value is None for value in values):
            raise ValueError('CA3_REFINEMENT_INPUT_CLOCK_COVERAGE_DIFFERS')
        first = values[0]
        if (first.start_ms, first.stop_ms) != (intervals, intervals+1):
            raise ValueError('CA3_REFINEMENT_INPUT_CLOCK_DIFFERS')
        for value in values:
            if (value.start_ms, value.stop_ms) != (first.start_ms, first.stop_ms):
                raise ValueError('CA3_REFINEMENT_INPUT_CLOCK_DIFFERS')
            # iter_intervals already binds R/A/C records and complete group IDs.
            for field in ('requested_current', 'effective_current'):
                current, reference = getattr(value, field), getattr(first, field)
                if (current.shape != (units,) or current.dtype != np.dtype('<f4')
                        or not np.array_equal(current.view('<u4'), reference.view('<u4'))):
                    raise ValueError('CA3_REFINEMENT_INPUT_BITS_DIFFER:' + str(intervals) + ':' + field)
        intervals += 1
    if intervals != DURATION_MS:
        raise ValueError('CA3_REFINEMENT_INPUT_CLOCK_COVERAGE_DIFFERS')
    return dict(intervals_compared=intervals, units=units, current_values_per_run=units*intervals,
                runs_compared=len(traces), requested_and_applied_bits_equal=True, tolerance=0,
                current_unit='pA', independent_reference_rng_replay_reused=True)


def comparison_pairs(counts):
    pairs = ((5, 10), (10, 20), (5, 20))
    return pairs + ((20, 40), (10, 40), (5, 40)) if 40 in counts else pairs


def phase_differences(names, sizes, counts):
    rows = []
    for coarse, fine in comparison_pairs(counts):
        for phase, start, stop in zip(PHASE_NAMES, PHASE_EDGES_MS[:-1], PHASE_EDGES_MS[1:]):
            for column, name in enumerate(names):
                a = int(counts[coarse][start:stop, column].sum())
                b = int(counts[fine][start:stop, column].sum())
                denominator = int(sizes[column])*(stop-start)/1000
                rows.append(dict(coarse_substeps=coarse, fine_substeps=fine, phase=phase,
                    field=name, units=int(sizes[column]), start_ms=start, stop_ms=stop,
                    coarse_count=a, fine_count=b, fine_minus_coarse_count=b-a,
                    coarse_rate_hz=a/denominator, fine_rate_hz=b/denominator,
                    fine_minus_coarse_rate_hz=(b-a)/denominator))
    return rows


def analyze(reference, ten, twenty, source_analysis, source_rng, output, *, input_origins=None, forty=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='RUNNING', scientific_fidelity_qualified=False,
                  numerical_precision_qualified=False, formation_refined=False,
                  whole_simulator_convergence_qualified=False, inferential_statistics_performed=False,
                  quantitative_tolerances_applied=False, complete_microstate_equality_observed=False,
                  independent_circuits=1, configured_initial_noise_blocks=1, production_episodes=0)
    try:
        runs, proofs = bind_runs(reference, ten, twenty, source_analysis, source_rng, forty=forty)
        units = runs[5].native['core_neurons']
        report['input_comparison'] = compare_inputs([run.root/'native_observation.bin' for run in runs.values()], units)
        fields, unit_counts, comparisons = {}, {}, {}
        names = sizes = None
        report['counter_checks'] = {}
        for count, run in runs.items():
            events, _ = _events(run, retain=True)
            checks = report['counter_checks'][str(count)] = {}
            _compare_counters(run.root/'native_observation.bin', events, units, checks)
            labels, group_sizes, fields[count] = field_counts(events, run.groups)
            if names is not None and (names != labels or not np.array_equal(sizes, group_sizes)):
                raise ValueError('CA3_REFINEMENT_FIELD_MAP_DIFFERS')
            names, sizes = labels, group_sizes
            unit_counts[count] = run.saved['unit_counts']
            comparisons[str(count)] = dict(events=len(events), native_elapsed_seconds=run.native['native_elapsed_seconds'],
                child_peak_rss_bytes=run.native['child_peak_rss_bytes'], integration=run.native['integration'],
                input_sha256=run.hashes)
        rows = phase_differences(names, sizes, fields)
        with (output/'phase_differences.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        differences = []
        for coarse, fine in comparison_pairs(runs):
            delta = unit_counts[fine] - unit_counts[coarse]
            differences.append(dict(coarse_substeps=coarse, fine_substeps=fine,
                units=units, changed_unit_totals=int(np.count_nonzero(delta)),
                all_unit_spike_count_mae=float(np.abs(delta).mean()),
                all_unit_spike_count_rmse=float(np.sqrt(np.mean(delta.astype(float)**2))),
                population_1ms_count_rmse=float(np.sqrt(np.mean((fields[fine][:, :8]-fields[coarse][:, :8]).astype(float)**2)))))
        np.savez_compressed(output/'counts.npz', substeps=np.asarray(list(runs)),
            unit_ids=runs[5].saved['unit_ids'], unit_totals=np.stack([unit_counts[n] for n in runs]),
            time_ms=np.arange(DURATION_MS), field_names=np.asarray(names), field_units=sizes,
            field_counts=np.stack([fields[n] for n in runs]))
        report.update(status='MEASURED_CA3_NEURON_SUBSTEP_SENSITIVITY', units=units,
            duration_ms=DURATION_MS, runs=comparisons, source_evidence=proofs,
            whole_schedule_differences=differences, phase_rows=len(rows),
            input_origins=input_origins or dict(reference=str(reference), ten=str(ten), twenty=str(twenty),
                                               source_analysis=str(source_analysis), source_rng=str(source_rng),
                                               **({'forty': str(forty)} if forty is not None else {})),
            interpretation='Fixed learned graph and matched1ms inputs; descriptive retrieval-output sensitivity only.',
            artifact_sha256={name:sha256_file(output/name) for name in ('phase_differences.csv','counts.npz')})
    except BaseException as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started, implementation_sha256=sha256_file(Path(__file__)))
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'ten', 'twenty', 'source-analysis', 'source-rng', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--input-origins', type=Path)
    parser.add_argument('--forty', type=Path, help='Optional matched40-substep retrieval')
    args = parser.parse_args()
    origins = json.loads(args.input_origins.read_text()) if args.input_origins else None
    print(json.dumps(analyze(args.reference, args.ten, args.twenty, args.source_analysis,
                             args.source_rng, args.output, input_origins=origins, forty=args.forty), indent=2))


if __name__ == '__main__':
    main()

"""Summarize the complete PFC coupled-clock readout without new simulations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def registered_blocks(protocol, report):
    """Keep the original pilot and the registered remaining bank disjoint."""
    starts = {'reference_2560': 93001, 'expanded_100k': 95001}
    if protocol is None:
        return {name: set(range(start, start+3)) for name, start in starts.items()}
    require(protocol['schema'] == 'pfc_coupled_clock_recurrent_gain_replicates_v1'
            and report['scope'] == 'development_recurrent_gain_coupled_clock_replicates_comparison'
            and protocol['independent_blocks_per_size'] == 29
            and protocol['full_existing_blocks_per_size'] == 32
            and protocol['analysis_partitions'] == ['pilot3', 'remaining29', 'full32_descriptive']
            and protocol['newly_selected_independent_source_blocks'] == 0
            and protocol['new_fine_source_episodes'] == 348
            and protocol['coarse_source_episodes_reused'] == 174
            and protocol['reused_pilot_fine_source_episodes'] == 36
            and protocol['confirmation_execution_released'] is False
            and protocol['scientific_fidelity_qualified'] is False,
            'registered remaining29 protocol required')
    expected = {name: set(range(start+3, start+32)) for name, start in starts.items()}
    require(set(protocol['cohorts']) == set(expected) and all(
        len(protocol['cohorts'][name]['source_rng_seeds']) == 29
        and set(protocol['cohorts'][name]['source_rng_seeds']) == seeds
        for name, seeds in expected.items()), 'complete disjoint remaining29 blocks required')
    return expected


def read_pilot_summary(archive, protocol):
    """Reuse only the accepted small pilot summaries bound to this protocol."""
    delivery = json.loads((archive/'delivery.json').read_text())
    require(delivery['status'] == 'COMPLETE' and delivery['exit_code'] == 0,
            'completed pilot summary delivery required')
    files = {row['path']: row for row in delivery['files']}
    bindings = {}

    def bound(name):
        path = archive/'summary'/name
        entry = files['summary/'+name]
        digest = sha(path)
        require(path.stat().st_size == entry['bytes'] and digest == entry['sha256'],
                'delivered pilot summary differs: '+name)
        bindings[name] = digest
        return path

    report = json.loads(bound('report.json').read_text())
    require(report['status'] == 'COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_SUMMARY'
            and report['complete_source_episodes'] == 54
            and report['fine_source_episodes'] == 36 and report['coarse_episodes_reused'] == 18
            and report['original_independent_blocks_per_size'] == 3
            and report['condition_pairing_rows_passed'] == 54
            and report['input_sha256']['analysis/report.json'] == protocol['reused_pilot_readout_sha256']
            and all(report[k] is False for k in ('scientific_fidelity_qualified',
                'confirmation_execution_released', 'confidence_intervals', 'p_values', 'cross_size_pairing')),
            'matching accepted pilot3 summary required')
    tables = {}
    for name, entry in report['outputs'].items():
        path = bound(name)
        require(entry['sha256'] == bindings[name] and entry['bytes'] == path.stat().st_size,
                'pilot report and delivery differ')
        with path.open(newline='') as stream:
            tables[name] = list(csv.DictReader(stream))
    return report, tables, dict(job_id=delivery['job_id'],
        delivery_sha256=sha(archive/'delivery.json'), input_sha256=bindings)


def partition_summaries(remaining, pilot, keys, metrics):
    """Pool disjoint blocks, retaining both partitions and between-group variance."""
    def index(rows, count):
        result = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            require(key not in result and int(row['blocks']) == count,
                    'duplicate endpoint or wrong partition block count')
            result[key] = row
        return result

    first, second = index(pilot, 3), index(remaining, 29)
    require(first and set(first) == set(second), 'identical pilot and remaining endpoint domains required')
    result = []
    for key in second:
        rows = [first[key], second[key]]
        pooled = dict(zip(keys, key), blocks=32)
        for metric in metrics:
            values = [{suffix: float(row[metric+'_'+suffix])
                       for suffix in ('mean', 'block_sd', 'min', 'max')} for row in rows]
            require(all(all(math.isfinite(x) for x in value.values())
                        and value['block_sd'] >= 0
                        and (value['min'] <= value['mean'] or math.isclose(value['min'], value['mean'], rel_tol=1e-12, abs_tol=1e-12))
                        and (value['mean'] <= value['max'] or math.isclose(value['mean'], value['max'], rel_tol=1e-12, abs_tol=1e-12))
                        for value in values), 'invalid partition moments')
            left, right = values
            mean = math.fsum((3*left['mean'], 29*right['mean']))/32
            m2 = math.fsum((2*left['block_sd']**2, 28*right['block_sd']**2,
                           (right['mean']-left['mean'])**2*3*29/32))
            pooled.update({metric+'_mean': mean, metric+'_block_sd': math.sqrt(m2/31),
                           metric+'_min': min(left['min'], right['min']),
                           metric+'_max': max(left['max'], right['max'])})
        result.extend([dict(partition='pilot3', **rows[0]),
                       dict(partition='remaining29', **rows[1]),
                       dict(partition='full32_descriptive', **pooled)])
    return result


def partition_block_rows(remaining, pilot, expected, keys):
    """Store each block once; full32 is the union, not another independent sample."""
    result = []
    for name, rows, blocks in (
        ('pilot3', pilot, registered_blocks(None, {})), ('remaining29', remaining, expected)):
        seen, identities = set(), set()
        for row in rows:
            cohort, seed = row['cohort'], int(row['source_rng_seed'])
            require(cohort in blocks and seed in blocks[cohort], 'wrong block partition')
            identity = (cohort, seed, *(str(row[k]) for k in keys))
            require(identity not in identities, 'duplicate per-block summary')
            seen.add((cohort, seed)); identities.add(identity)
            result.append(dict(partition=name, **row))
        require(seen == {(cohort, seed) for cohort, seeds in blocks.items() for seed in seeds},
                'complete partition block coverage required')
    return result


def summarize(source, output, *, protocol_path=None, pilot_summary=None):
    require((protocol_path is None) == (pilot_summary is None),
            'remaining29 protocol and accepted pilot summary must be supplied together')
    protocol = yaml.safe_load(protocol_path.read_text()) if protocol_path else None
    delivery = json.loads((source/'delivery.json').read_text())
    files = {row['path']: row for row in delivery['files']}
    bindings = {}

    def bound(name):
        path = source/name
        digest = sha(path)
        require(path.stat().st_size == files[name]['bytes'] and digest == files[name]['sha256'],
                'delivered input differs: ' + name)
        bindings[name] = digest
        return path

    wrapper = json.loads(bound('report.json').read_text())
    report = json.loads(bound('analysis/report.json').read_text())
    expected = registered_blocks(protocol, report)
    blocks_per_size = 29 if protocol is not None else 3
    block_count = 2*blocks_per_size
    if protocol is not None:
        require(report['protocol_sha256'] == sha(protocol_path), 'source protocol differs')
    require(delivery['verified'] is True and wrapper['exit_code'] == 0
            and report['status'] == delivery['status'] == 'COMPLETE_DESCRIPTIVE_RECURRENT_GAIN_CLOCK_COMPARISON'
            and report['complete_source_episodes_consumed'] == 9*block_count
            and report['coarse_episodes_reused'] == 3*block_count
            and report['coupled_fine_episodes_consumed'] == 6*block_count
            and report['source_simulations'] == report['production_episodes'] == 0
            and report['source_evidence_kind'] == 'archived_source_execution'
            and all(report[k] is False for k in ('cross_size_pairing', 'confidence_intervals',
                'scientific_fidelity_qualified', 'confirmation_execution_released')),
            'complete descriptive registered source readout required')
    require(len(report['blocks']) == block_count and {(b['cohort'], b['source_rng_seed']) for b in report['blocks']}
            == {(cohort, seed) for cohort, seeds in expected.items() for seed in seeds},
            'all registered blocks required')
    pilot = read_pilot_summary(pilot_summary, protocol) if protocol is not None else None
    groups = defaultdict(dict)
    cdf_points, unit_rows = [], []
    consumed = defaultdict(int)
    shared_checks = 0
    common = ['cohort', 'left_clock', 'right_clock', 'condition', 'phase']
    specifications = {
        'source_response_comparison.csv': (common+['metric', 'units'], ['left_value', 'right_value', 'left_minus_right']),
        'paired_response_comparison.csv': ([x if x != 'condition' else 'contrast' for x in common]+['metric', 'units'],
                                           ['left_value', 'right_value', 'left_minus_right']),
        'field_clock_discrepancies.csv': (common+['field_index', 'units'],
                                          ['signed_mean_hz', 'mean_absolute_hz', 'rms_hz', 'maximum_absolute_hz']),
    }
    group_identity = {}
    for block in report['blocks']:
        cohort, seed = block['cohort'], block['source_rng_seed']
        prefix = f'analysis/{cohort}/{seed}/'

        def rows(name):
            with bound(prefix+name).open(newline='') as stream:
                result = list(csv.DictReader(stream))
            count_key = name.removesuffix('.csv') if name in ('shared_input_checks.csv', 'semantic_phase_readouts.csv') else name
            require(len(result) == block['row_counts'][count_key], 'complete table rows required')
            require(all(r['cohort'] == cohort and int(r['source_rng_seed']) == seed
                        and r['evidence_kind'] == 'archived_source_execution' for r in result), 'row identity differs')
            consumed[count_key] += len(result)
            return result

        for name, (keys, metrics) in specifications.items():
            for row in rows(name):
                identity = {key: row[key] for key in keys}
                values = {key: float(row[key]) for key in metrics}
                require(all(np.isfinite(v) for v in values.values()), 'nonfinite summarized metric')
                if 'left_minus_right' in values:
                    require(np.isclose(values['left_value']-values['right_value'], values['left_minus_right'],
                                       rtol=1e-10, atol=1e-9), 'saved signed difference differs')
                key = (name, *identity.values())
                require(seed not in groups[key], 'duplicate block row')
                groups[key][seed] = values
                group_identity[key] = identity
        for row in rows('shared_input_checks.csv'):
            require(all(row[k] == 'True' for k in ('exact_pre_cue_events', 'exact_initial_voltage',
                        'exact_final_source_rng', 'same_source_input_consumption')), 'condition pairing failed')
            shared_checks += 1
        rows('semantic_phase_readouts.csv')
        curves = defaultdict(list)
        keys = common+['window_ms', 'field_index']
        for row in rows('count_cdf_comparison.csv'):
            key = tuple(row[k] for k in keys)
            left, right, difference = (float(row[k]) for k in ('left_cdf', 'right_cdf', 'left_minus_right'))
            require(0 <= left <= 1 and 0 <= right <= 1 and np.isclose(left-right, difference, rtol=1e-10, atol=1e-12),
                    'invalid CDF value or difference')
            curves[key].append((int(row['count_threshold']), left, right, difference))
        for key, curve in curves.items():
            curve.sort()
            require([r[0] for r in curve] == list(range(len(curve))) and curve[-1][1:3] == (1., 1.)
                    and all(a[1] <= b[1] and a[2] <= b[2] for a, b in zip(curve, curve[1:])),
                    'complete monotone CDF support required')
            peak = max(curve, key=lambda r: abs(r[3]))
            cdf_points.append(dict(zip(keys, key), source_rng_seed=seed,
                maximum_absolute_cdf_difference=abs(peak[3]), signed_difference_at_first_maximum=peak[3],
                first_maximizing_count_threshold=peak[0], maximum_saved_count_threshold=curve[-1][0]))
        with np.load(bound(prefix+'unit_count_clock_differences.npz'), allow_pickle=False) as stored:
            require(stored['source_rng_seeds'].tolist() == [seed] and stored['cohort_names'].tolist() == [cohort]
                    and stored['evidence_kind'].item() == 'archived_source_execution'
                    and stored['bin_metric_names'].tolist() == ['signed_mean_count', 'mean_absolute_count', 'rms_count'],
                    'unit difference identities or metric axes differ')
            phase = stored['phase_count_left_minus_right'][0, 0]
            bins = stored['bin_count_differences'][0, 0]
            total = 2560 if cohort == 'reference_2560' else 100000
            require(phase.shape == (3, 3, 5, total) and bins.shape == (3, 3, 5, 3, total)
                    and np.array_equal(stored['source_unit_ids'], np.arange(total)), 'all unit outputs required')
            require(np.isfinite(bins).all() and np.all(bins[:, :, :, 1:] >= 0), 'invalid unit error values')
            phase_bins = stored['matched_phase_bins']
            require(np.allclose(bins[:, :, :, 0] * phase_bins[None, None, :, None], phase,
                                rtol=1e-10, atol=1e-9), 'phase totals and signed bin means differ')
            for pi, pair in enumerate(stored['clock_pairs'].tolist()):
                for ci, condition in enumerate(stored['condition_names'].tolist()):
                    for wi, window in enumerate(stored['phase_names'].tolist()):
                        for pop, mask in [('all', np.ones(total, dtype=bool)),
                                          ('E', stored['population_indices'] == 0),
                                          ('I', stored['population_indices'] == 1)]:
                            delta = phase[pi, ci, wi, mask]
                            unit_rows.append(dict(cohort=cohort, source_rng_seed=seed, left_clock=pair[0],
                                right_clock=pair[1], condition=condition, phase=window, population=pop,
                                units=int(mask.sum()), units_with_different_phase_count=int(np.count_nonzero(delta)),
                                phase_count_changed_fraction=float(np.count_nonzero(delta)/mask.sum()),
                                mean_signed_phase_count=float(delta.mean()), mean_absolute_phase_count=float(np.abs(delta).mean()),
                                mean_unit_bin_absolute_count=float(bins[pi, ci, wi, 1, mask].mean()),
                                root_mean_square_unit_bin_count=float(np.sqrt(np.square(bins[pi, ci, wi, 2, mask]).mean()))))
    require(dict(consumed) == report['row_counts'] and shared_checks == 9*block_count,
            'complete source table coverage required')
    summaries = defaultdict(list)
    for key, blocks in groups.items():
        identity = group_identity[key]
        require(set(blocks) == expected[identity['cohort']], 'all registered blocks required per summary')
        row = dict(identity, blocks=blocks_per_size)
        for metric in next(iter(blocks.values())):
            values = [blocks[seed][metric] for seed in sorted(blocks)]
            row.update({metric+'_mean': statistics.fmean(values), metric+'_block_sd': statistics.stdev(values),
                        metric+'_min': min(values), metric+'_max': max(values)})
        summaries[key[0]].append(row)
    if pilot is not None:
        pilot_report, pilot_tables, pilot_binding = pilot
        for name, (keys, metrics) in specifications.items():
            table = pilot_tables[name.removesuffix('.csv')+'_summary.csv']
            require(len(table) == pilot_report['summary_rows'][name], 'complete pilot endpoint rows required')
            summaries[name] = partition_summaries(summaries[name], table, keys, metrics)
        pilot_cdf = pilot_tables['count_cdf_suprema_by_block.csv']
        pilot_units = pilot_tables['unit_differences_by_block.csv']
        require(len(pilot_cdf) == pilot_report['cdf_supremum_rows']
                and len(pilot_units) == pilot_report['unit_difference_rows'], 'complete pilot block rows required')
        cdf_points = partition_block_rows(cdf_points, pilot_cdf, expected,
                                         common[1:]+['window_ms', 'field_index'])
        unit_rows = partition_block_rows(unit_rows, pilot_units, expected, common[1:]+['population'])
    output.mkdir(parents=True, exist_ok=False)
    names = []
    for name, table in summaries.items():
        target = name.removesuffix('.csv')+'_summary.csv'
        write_rows(output/target, table)
        names.append(target)
    for name, table in [('count_cdf_suprema_by_block.csv', cdf_points), ('unit_differences_by_block.csv', unit_rows)]:
        write_rows(output/name, table)
        names.append(name)
    result = dict(status='COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_SUMMARY', recorded_utc=datetime.now(timezone.utc).isoformat(),
        source_job=delivery['job_id'], source_delivery_sha256=sha(source/'delivery.json'), input_sha256=bindings,
        complete_source_episodes=9*block_count, fine_source_episodes=6*block_count,
        coarse_episodes_reused=3*block_count,
        original_independent_blocks_per_size=blocks_per_size, condition_pairing_rows_passed=shared_checks,
        input_csv_rows=dict(consumed), summary_rows={name:len(rows) for name,rows in summaries.items()},
        cdf_supremum_rows=len(cdf_points), unit_difference_rows=len(unit_rows),
        unit_diff_npz_files_read=block_count, raw_source_arrays_read=0, source_simulations=0, refits=0,
        scientific_fidelity_qualified=False, confirmation_execution_released=False, production_episodes=0,
        confidence_intervals=False, p_values=False, cross_size_pairing=False,
        interpretation='Three exposed source RNG blocks per size; conditions, units and time bins are not independent replicates. '
            'Block SD is descriptive. CDF suprema summarize empirical count distributions and are not significance tests. '
            'Coupled binomial marginal laws do not share an identical event list; differences are not pure integration error. '
            'This pilot does not validate the distinct new29 weak-response distribution or qualify convergence/production.',
        implementation_sha256=sha(Path(__file__)), outputs={name:dict(bytes=(output/name).stat().st_size,
        sha256=sha(output/name)) for name in names})
    if pilot is not None:
        result.update(complete_source_episodes=576, fine_source_episodes=384, coarse_episodes_reused=192,
            original_independent_blocks_per_size=32, condition_pairing_rows_passed=shared_checks+54,
            newly_consumed_source_episodes=522, pilot_source_episodes_reused_via_summary=54,
            condition_pairing_rows_newly_checked=shared_checks, condition_pairing_rows_reused=54,
            remaining_independent_blocks_per_size=29, reused_pilot_summary=pilot_binding,
            protocol_sha256=sha(protocol_path), partitions=['pilot3', 'remaining29', 'full32_descriptive'],
            interpretation='All 32 existing source RNG blocks per size; pilot3 and remaining29 are disjoint, '
                'and full32_descriptive is their union, not an independent comparison. Pilot moments and '
                'per-block summaries are reused without raw pilot arrays. Conditions, units and time bins '
                'are not independent replicates; block SD is descriptive. Empirical CDF suprema are not '
                'significance tests. Coupled binomial marginal laws are not identical event lists, so '
                'differences are not pure integration error. No convergence, prevalence, cross-gain '
                'interaction, parameter-domain, confirmation or production qualification is established.')
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return {key:result[key] for key in ('status', 'input_csv_rows', 'summary_rows', 'cdf_supremum_rows', 'unit_difference_rows')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        help='Registered remaining29 source protocol; requires --pilot-summary')
    parser.add_argument('--pilot-summary', type=Path,
                        help='Accepted pilot summary archive, reused without raw pilot arrays')
    args = parser.parse_args()
    print(json.dumps(summarize(args.source, args.output, protocol_path=args.protocol,
                               pilot_summary=args.pilot_summary), indent=2))


if __name__ == '__main__':
    main()

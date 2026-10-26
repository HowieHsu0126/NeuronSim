"""Read every delivered PFC precision endpoint without fitting or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from .provenance import sha256_file


def readback(source: Path, output: Path) -> dict:
    delivery = json.loads((source / 'delivery.json').read_text())
    if delivery['status'] != 'COMPLETE_PFC_COUPLED_PRECISION_COUPLED' or delivery['verified'] is not True:
        raise ValueError('complete delivered coupled comparison required')
    entries = {row['path']: row for row in delivery['files']}
    if len(entries) != len(delivery['files']):
        raise ValueError('unique delivery paths required')
    names = ('report.json', 'analysis/report.json', 'analysis/endpoint_precision_comparison.csv')
    for name in names:
        path, entry = source / name, entries[name]
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError('delivered input changed: ' + name)
    execution = json.loads((source / 'report.json').read_text())
    report = json.loads((source / 'analysis/report.json').read_text())
    if (execution['mode'] != 'coupled' or execution['exit_code'] != 0
            or report['status'] != 'COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON'
            or report['original_independent_blocks'] != 32
            or report['confirmation_execution_released'] is not False):
        raise ValueError('completed fixed-bank descriptive comparison required')
    name = 'endpoint_precision_comparison.csv'
    if report['outputs'][name] != {key: entries['analysis/' + name][key] for key in ('bytes', 'sha256')}:
        raise ValueError('analysis and delivery table bindings differ')

    total, by_family, definitions = Counter(), {}, Counter()
    statuses = {arm: Counter() for arm in ('native', 'coupled')}
    tails = {arm: Counter() for arm in statuses}
    ratios = Counter()
    identities, increases, exceedances = set(), [], []
    cost = report['observed_simulator_cost_ratio']
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError('positive measured cost ratio required')
    with (source / 'analysis' / name).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        for row in reader:
            identity = int(row['domain_row_index'])
            if identity in identities:
                raise ValueError('duplicate frozen domain row')
            identities.add(identity)
            counts = by_family.setdefault(row['family'], Counter())

            def count(key):
                total[key] += 1
                counts[key] += 1

            count('rows')
            definitions[row['definition_status']] += 1
            ratios[row['variance_ratio_status']] += 1
            for arm in statuses:
                statuses[arm][row[arm + '_empirical_status']] += 1
                tails[arm][row[arm + '_cdf_tail_status'] or 'NOT_APPLICABLE'] += 1
            if row['variance_ratio_status'] == 'UNDEFINED':
                if any(row[key] != '' for key in ('margin', 'native_signed_estimate', 'coupled_signed_estimate')):
                    raise ValueError('undefined endpoint gained a numeric estimate')
                count('undefined')
                continue
            count('defined')
            margin = float(row['margin'])
            if not math.isfinite(margin) or margin < 0:
                raise ValueError('finite nonnegative frozen margin required')
            for arm in statuses:
                estimate, variance, se = (float(row[arm + '_' + key]) for key in
                                          ('signed_estimate', 'sample_variance', 'standard_error'))
                if not all(math.isfinite(x) for x in (estimate, variance, se)) or variance < 0 or se < 0:
                    raise ValueError('finite estimate and nonnegative empirical uncertainty required')
                if int(row[arm + '_original_independent_blocks']) != 32:
                    raise ValueError('original block denominator changed')
                if not math.isclose(se * se * 32, variance, rel_tol=1e-12, abs_tol=1e-30):
                    raise ValueError('standard error and block variance disagree')
                count(arm + ('_point_outside_margin' if abs(estimate) > margin else '_point_within_margin'))
                if arm == 'coupled' and abs(estimate) > margin:
                    exceedances.append(row)
                if variance == 0:
                    count(arm + '_zero_empirical_variance')
            native, coupled = float(row['native_sample_variance']), float(row['coupled_sample_variance'])
            change = 'lower' if coupled < native else 'higher' if coupled > native else 'equal'
            count('variance_' + change)
            if change == 'higher':
                increases.append(row)
            if float(row['observed_simulator_cost_ratio']) != cost:
                raise ValueError('endpoint cost ratio differs from measured report')
            if row['variance_ratio_status'] == 'DEFINED_RATIO':
                ratio = float(row['variance_ratio'])
                adjusted = float(row['variance_simulator_cost_ratio'])
                if native <= 0 or not math.isclose(ratio, coupled / native, rel_tol=1e-12, abs_tol=1e-30):
                    raise ValueError('variance ratio differs from endpoint variances')
                if not math.isclose(adjusted, ratio * cost, rel_tol=1e-12, abs_tol=1e-30):
                    raise ValueError('variance-cost proxy differs from declared components')
                count('variance_cost_' + ('lower' if adjusted < 1 else 'higher' if adjusted > 1 else 'equal'))
            else:
                if row['variance_ratio'] != '' or row['variance_simulator_cost_ratio'] != '':
                    raise ValueError('undefined variance ratios must remain empty')
                count('variance_cost_ratio_unavailable')

    if total['rows'] != report['row_count'] or any(total['variance_' + k] != value
                                                for k, value in report['variance_change_rows'].items()):
        raise ValueError('full-table readback differs from producer row counts')
    result = dict(status='DESCRIBED_COMPLETE_FROZEN_ENDPOINT_COMPARISON',
        source_delivery_sha256=sha256_file(source / 'delivery.json'),
        source_report_sha256=entries['analysis/report.json']['sha256'],
        source_table_sha256=entries['analysis/endpoint_precision_comparison.csv']['sha256'],
        implementation_sha256=sha256_file(Path(__file__)),
        source_job=execution['job_id'], original_independent_blocks=32,
        independent_circuit_instances=report['independent_circuit_instances'],
        counts=dict(total), by_family={k: dict(v) for k, v in by_family.items()},
        definition_status_counts=dict(definitions), variance_ratio_status_counts=dict(ratios),
        empirical_status_counts={k: dict(v) for k, v in statuses.items()},
        cdf_tail_status_counts={k: dict(v) for k, v in tails.items()},
        observed_simulator_cost_ratio=cost, cost_measure=report['cost_measure'],
        all_variance_increases_retained=len(increases),
        all_coupled_point_exceedances_retained=len(exceedances), confidence_intervals=False,
        numerical_margins_changed=False, source_simulations=0, production_episodes=0,
        scientific_fidelity_qualified=False, uncertainty_coverage_qualified=False,
        confirmation_execution_released=False, matched_end_to_end_speedup_qualified=False,
        interpretation='All frozen rows retained in the source table. Point containment and empirical variance-cost '
        'are descriptive; they do not establish equivalence, tail coverage, or confirmation feasibility. '
        'Empirical zero is not population zero. No selection, exclusion, resampling or interval construction.')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('variance_increases.csv', increases),
                       ('coupled_points_outside_margin.csv', exceedances)):
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def diagnose_blocks(source: Path, parent: Path, output: Path) -> dict:
    """Trace every previously identified exceedance, preserving all 32 blocks."""
    source_delivery = json.loads((source / 'delivery.json').read_text())
    parent_delivery = json.loads((parent / 'delivery.json').read_text())
    if (source_delivery['status'] != 'COMPLETE_PFC_COUPLED_PRECISION_COUPLED'
            or source_delivery['verified'] is not True
            or parent_delivery['status'] != 'COMPLETE_PFC_PRECISION_READBACK'
            or parent_delivery['exit_code'] != 0
            or parent_delivery['source_delivery_sha256'] != sha256_file(source / 'delivery.json')):
        raise ValueError('completed readback of the same coupled comparison required')
    for root, delivery, names in (
            (source, source_delivery, ('analysis/report.json', 'analysis/coupled_block_scores.npz')),
            (parent, parent_delivery, ('analysis/report.json', 'analysis/coupled_points_outside_margin.csv'))):
        entries = {row['path']: row for row in delivery['files']}
        if len(entries) != len(delivery['files']):
            raise ValueError('unique delivered input paths required')
        for name in names:
            path, entry = root / name, entries[name]
            if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
                raise ValueError('delivered block-diagnosis input changed: ' + name)
    producer = json.loads((source / 'analysis/report.json').read_text())
    previous = json.loads((parent / 'analysis/report.json').read_text())
    if (previous['source_report_sha256'] != sha256_file(source / 'analysis/report.json')
            or previous['source_delivery_sha256'] != sha256_file(source / 'delivery.json')
            or previous['counts']['rows'] != producer['row_count']):
        raise ValueError('parent full-domain readback binding changed')
    score_path = source / 'analysis/coupled_block_scores.npz'
    if producer['outputs']['coupled_block_scores.npz']['sha256'] != sha256_file(score_path):
        raise ValueError('score payload differs from producer analysis')
    with (parent / 'analysis/coupled_points_outside_margin.csv').open(newline='') as stream:
        selected = list(csv.DictReader(stream))
    if (len(selected) != previous['counts']['coupled_point_outside_margin']
            or len(selected) != previous['all_coupled_point_exceedances_retained']):
        raise ValueError('all previously identified exceedances must be retained')
    with np.load(score_path, allow_pickle=False) as data:
        seeds, indices, scores = data['source_rng_seeds'], data['domain_row_indices'], data['scores']
        if (indices.dtype.kind not in 'iu' or seeds.dtype.kind not in 'iu'
                or len(set(indices.tolist())) != len(indices)
                or len(indices) != producer['row_count'] or len(seeds) != 32
                or not np.array_equal(seeds, producer['settings']['source_rng_seeds'])
                or scores.shape != (len(indices), len(seeds))):
            raise ValueError('complete frozen row and original block identities required')
        lookup = {int(value): index for index, value in enumerate(indices)}
        contributions, endpoints, paired = [], [], {}
        seen = set()
        for row in selected:
            identity = int(row['domain_row_index'])
            if identity in seen or row['family'] not in ('field_mean', 'paired_response'):
                raise ValueError('distinct linear residual endpoints required')
            seen.add(identity)
            values = np.asarray(scores[lookup[identity]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('finite complete residual block scores required')
            mean, variance = float(values.mean()), float(values.var(ddof=1))
            se = math.sqrt(variance / len(seeds))
            deviations = values - mean
            energy = float(np.dot(deviations, deviations))
            fractions = deviations ** 2 / energy if energy > 0 else np.zeros_like(values)
            influence = float(fractions.max())
            for name, actual in (('signed_estimate', mean), ('sample_variance', variance),
                                 ('standard_error', se), ('maximum_block_influence_fraction', influence)):
                if not math.isclose(actual, float(row['coupled_' + name]), rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError('residual block scores do not reproduce recorded ' + name)
            if int(np.count_nonzero(values)) != int(row['coupled_nonzero_score_blocks']):
                raise ValueError('nonzero block count changed')
            margin = float(row['margin'])
            if abs(mean) <= margin:
                raise ValueError('parent residual does not exceed its frozen margin')
            leave_one_out = (values.sum() - values) / (len(values) - 1)
            endpoint = json.loads(row['endpoint_identity'])
            endpoints.append(dict(domain_row_index=identity, family=row['family'], endpoint_identity=endpoint,
                units=row['units'], margin=margin, mean=mean, standard_error=se,
                nonzero_score_blocks=int(np.count_nonzero(values)),
                positive_score_blocks=int(np.count_nonzero(values > 0)),
                negative_score_blocks=int(np.count_nonzero(values < 0)),
                maximum_influence_source_rng_seed=int(seeds[int(np.argmax(fractions))]),
                maximum_block_influence_fraction=influence,
                minimum_leave_one_block_out_mean=float(leave_one_out.min()),
                maximum_leave_one_block_out_mean=float(leave_one_out.max()),
                leave_one_block_out_points_outside_margin=int(np.count_nonzero(np.abs(leave_one_out) > margin))))
            for seed, value, fraction, omitted_mean in zip(seeds, values, fractions, leave_one_out):
                contributions.append(dict(domain_row_index=identity, family=row['family'],
                    source_rng_seed=int(seed), score=float(value), contribution_to_mean=float(value / len(seeds)),
                    squared_deviation_fraction=float(fraction), leave_one_block_out_mean=float(omitted_mean),
                    units=row['units'], frozen_margin=margin))
            if row['family'] == 'paired_response':
                group = (endpoint['contrast'], endpoint['phase'], endpoint['metric'])
                pair = (endpoint['left_cohort'], endpoint['right_cohort'])
                paired.setdefault(group, {})[pair] = (identity, values.copy())
    shared_clock = []
    for group, pairs in paired.items():
        left = pairs.get(('numpy_dt_0p05', 'numpy_dt_0p025'))
        right = pairs.get(('numpy_dt_0p025', 'numpy_dt_0p0125'))
        if left is None or right is None:
            continue
        combined = left[1] + right[1]
        shared_clock.append(dict(contrast=group[0], phase=group[1], metric=group[2],
            domain_row_indices=[left[0], right[0]],
            empirical_adjacent_score_correlation=float(np.corrcoef(left[1], right[1])[0, 1]),
            telescoped_point_estimate=float(combined.mean()),
            interpretation='Same-block algebraic sum around the shared middle clock; descriptive diagnosis, not a new qualification endpoint.'))
    result = dict(status='DESCRIBED_RESIDUAL_BLOCK_CONTRIBUTIONS',
        source_delivery_sha256=sha256_file(source / 'delivery.json'),
        parent_readback_delivery_sha256=sha256_file(parent / 'delivery.json'),
        source_block_scores_sha256=sha256_file(score_path), implementation_sha256=sha256_file(Path(__file__)),
        counts=dict(residual_endpoints=len(selected), original_independent_blocks=len(seeds),
                    endpoint_block_rows=len(contributions)), endpoints=endpoints, shared_middle_clock=shared_clock,
        source_rng_seeds=seeds.tolist(), source_simulations=0, production_episodes=0,
        excluded_blocks=0, leave_one_out_used_for_selection=False, confidence_intervals=False,
        scientific_fidelity_qualified=False, uncertainty_coverage_qualified=False,
        confirmation_execution_released=False, numerical_margins_changed=False,
        interpretation='All residuals and all32 original blocks retained. Omission values describe influence only; '
        'the complete32-block estimates remain authoritative. No inferential equivalence or inequivalence claim.')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'block_contributions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contributions[0]))
        writer.writeheader()
        writer.writerows(contributions)
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-readback', type=Path)
    args = parser.parse_args()
    result = (readback(args.source, args.output) if args.parent_readback is None
              else diagnose_blocks(args.source, args.parent_readback, args.output))
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()

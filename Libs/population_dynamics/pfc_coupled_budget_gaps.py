"""Classify saved native/coupled calibration budget gaps without new inference.

A negative signed side gap is slack inside that side of the frozen budget.
Projected bands are planning scenarios; neither transitions nor tail-value
coincidences establish coverage, numerical fidelity or a cause of a gap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path
import time

import yaml

from .pfc_coupled_precision_summary import require, _write_csv
from .provenance import freeze_json, sha256_file

SCHEMA = 'pfc_coupled_budget_gap_diagnosis_v1'
STRUCTURAL = 'STRUCTURAL_ZERO_CONTRAST'
UNDEFINED = 'UNDEFINED_NO_WITHIN_PHASE_PAIRS'
EXCEEDS = 'PLANNING_BAND_EXCEEDS_BUDGET'
WITHIN = 'PLANNING_BAND_WITHIN_BUDGET'
IDENTITY = ('family', 'endpoint_identity', 'definition_status')
FALSE_FLAGS = ('coverage_qualified', 'scientific_fidelity_qualified', 'confirmation_execution_released',
               'method_adopted_for_confirmation', 'uncertainty_coverage_qualified')
FILES = {'coupled': ['calibration_budget_assessment.csv', 'endpoint_bands.csv'],
         'native': ['calibration_budget_diagnosis.csv'],
         'native_bands': ['endpoint_bands.csv'], 'budget': ['frozen_numerical_margins.csv']}


def interval_relation(lower, upper, margin):
    require(all(math.isfinite(x) for x in (lower, upper, margin)) and margin > 0 and lower <= upper,
            'FINITE_ORDERED_INTERVAL_AND_POSITIVE_MARGIN_REQUIRED')
    if lower >= -margin and upper <= margin: return 'WITHIN_BUDGET'
    if upper < -margin or lower > margin: return 'DISJOINT_FROM_BUDGET'
    return 'OVERLAPS_BUDGET'


def _saved_values(assessment, band, definition):
    require(assessment['calibration_estimate'] == band['estimate']
            and assessment['planning_lower'] == band['planning_lower_root_n_and_zero_tail_at_128']
            and assessment['planning_upper'] == band['planning_upper_root_n_and_zero_tail_at_128'],
            'SAVED_ASSESSMENT_AND_BAND_VALUES_DIFFER')
    if definition == UNDEFINED:
        require(assessment['planning_status'] == 'UNDEFINED'
                and all(value == '' for value in [assessment['margin'], band['estimate'], band['lower'], band['upper'],
                                                  assessment['planning_lower'], assessment['planning_upper']]),
                'UNDEFINED_ENDPOINT_MUST_REMAIN_UNESTIMATED')
        return dict(estimate='', calibration_lower='', calibration_upper='', projected_lower='', projected_upper='',
                    planning_status='UNDEFINED', calibration_relation='UNDEFINED', projected_relation='UNDEFINED',
                    point_outside='', projected_exceeds=False, projected_disjoint=False,
                    point_positive_side_gap='', point_negative_side_gap='', calibration_positive_side_gap='',
                    calibration_negative_side_gap='', projected_positive_side_gap='', projected_negative_side_gap='',
                    projected_width='', projected_center='')
    delta, point, lo, hi, future_lo, future_hi = map(float, [assessment['margin'], band['estimate'], band['lower'],
                                                         band['upper'], assessment['planning_lower'], assessment['planning_upper']])
    if definition == STRUCTURAL:
        require(delta == point == lo == hi == future_lo == future_hi == 0 and assessment['planning_status'] == 'SOURCE_IDENTITY',
                'FROZEN_SOURCE_IDENTITY_CHANGED')
        actual_relation = future_relation = 'SOURCE_IDENTITY'; outside = False
    else:
        actual_relation = interval_relation(lo, hi, delta); future_relation = interval_relation(future_lo, future_hi, delta)
        outside = abs(point) > delta
        require(assessment['calibration_interval_relation'] == actual_relation
                and assessment['planning_interval_relation'] == future_relation
                and assessment['planning_status'] == (WITHIN if future_relation == 'WITHIN_BUDGET' else EXCEEDS),
                'SAVED_INTERVAL_CLASSIFICATION_CHANGED')
        require(float(assessment['planning_absolute_bound_over_margin']) == max(abs(future_lo), abs(future_hi))/delta,
                'SAVED_NORMALIZED_PROJECTED_BOUND_CHANGED')
    require(assessment['calibration_point_outside_margin'] == str(outside), 'SAVED_POINT_OUTSIDE_FLAG_CHANGED')
    return dict(estimate=point, calibration_lower=lo, calibration_upper=hi, projected_lower=future_lo, projected_upper=future_hi,
                planning_status=assessment['planning_status'], calibration_relation=actual_relation, projected_relation=future_relation,
                point_outside=outside, projected_exceeds=assessment['planning_status'] == EXCEEDS,
                projected_disjoint=future_relation == 'DISJOINT_FROM_BUDGET',
                point_positive_side_gap=point-delta, point_negative_side_gap=-delta-point,
                calibration_positive_side_gap=hi-delta, calibration_negative_side_gap=-delta-lo,
                projected_positive_side_gap=future_hi-delta, projected_negative_side_gap=-delta-future_lo,
                projected_width=future_hi-future_lo, projected_center=(future_lo+future_hi)/2)


def merge_row(index, margin, native, native_band, coupled, coupled_band, tail):
    rows = (margin, native, native_band, coupled, coupled_band)
    require(all(int(row['domain_row_index']) == index for row in rows)
            and all(all(row[key] == margin[key] for key in IDENTITY) for row in rows),
            'EXACT_ORDERED_FROZEN_ENDPOINT_JOIN_REQUIRED')
    require(native['margin'] == coupled['margin'] == margin['margin'], 'FROZEN_NUMERICAL_MARGIN_CHANGED')
    require(coupled_band['frozen_native_estimate'] == native_band['estimate'], 'FROZEN_NATIVE_ESTIMATE_CHANGED')
    ident = json.loads(margin['endpoint_identity']); require(isinstance(ident, dict), 'ENDPOINT_IDENTITY_REQUIRED')
    definition = margin['definition_status']; current = coupled_band['current_empirical_status']
    require((definition == STRUCTURAL) == (current == STRUCTURAL)
            and (definition == UNDEFINED) == (current == UNDEFINED), 'CURRENT_SOURCE_IDENTITY_OR_LAG_STATUS_CHANGED')
    result = dict(domain_row_index=index, **{k: margin[k] for k in IDENTITY}, margin=margin['margin'], units=margin['units'],
                  coupled_current_empirical_status=current, native_channel=native_band['channel'], coupled_channel=coupled_band['channel'])
    values = {}
    for prefix, assessment, band in [('native', native, native_band), ('coupled', coupled, coupled_band)]:
        values[prefix] = _saved_values(assessment, band, definition)
        result.update({prefix+'_'+key: value for key, value in values[prefix].items()})
    require(math.isfinite(tail) and 0 < tail < 1, 'FINITE_SAVED_PLANNING_TAIL_BOUND_REQUIRED')
    result['coupled_cdf_uses_primitive_channel'] = margin['family'] == 'count_cdf' and 'primitive_cdf' in coupled_band['channel']
    result['coupled_cdf_projected_absolute_bound_matches_saved_tail_value'] = (
        margin['family'] == 'count_cdf' and definition not in (STRUCTURAL, UNDEFINED)
        and math.isclose(max(abs(values['coupled']['projected_lower']), abs(values['coupled']['projected_upper'])), tail,
                         rel_tol=1e-10, abs_tol=1e-14))
    result['coupled_current_empirically_degenerate'] = current == 'EMPIRICALLY_DEGENERATE_UNCERTAIN'
    result['native_to_coupled_estimate_change'] = (values['coupled']['estimate']-values['native']['estimate']
                                                  if definition != UNDEFINED else '')
    return result


def summarize_tables(tables, expected_rows, tail, output):
    """Stream a complete ordered join and keep every adverse row."""
    transitions = {key: Counter() for key in ('planning_status', 'calibration_relation', 'projected_relation', 'point_outside')}
    counts = {prefix: {key: Counter() for key in transitions} for prefix in ('native', 'coupled')}
    families = defaultdict(Counter); definitions = Counter(); current = Counter()
    adverse, points, disjoint = [], [], []; count = 0
    with (output/'all_endpoint_budget_transitions.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = None
        for index, entries in enumerate(zip_longest(*tables)):
            require(all(row is not None for row in entries), 'MISSING_OR_EXTRA_ENDPOINT_TABLE_ROWS')
            row = merge_row(index, *entries, tail)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); count += 1
            definitions[row['definition_status']] += 1; current[row['coupled_current_empirical_status']] += 1
            for key in transitions:
                transitions[key][str(row['native_'+key]), str(row['coupled_'+key])] += 1
                for prefix in counts: counts[prefix][key][str(row[prefix+'_'+key])] += 1
            families[row['family']]['all_endpoints'] += 1
            for flag, subset in [('projected_exceeds', adverse), ('point_outside', points), ('projected_disjoint', disjoint)]:
                if row['coupled_'+flag] is True:
                    subset.append(row); families[row['family']][flag] += 1
        require(count == expected_rows, 'COMPLETE_FROZEN_ENDPOINT_DENOMINATOR_REQUIRED')
    require(writer is not None, 'NONEMPTY_FROZEN_DOMAIN_REQUIRED')
    for name, rows in [('coupled_projected_exceedances.csv', adverse), ('coupled_point_outside.csv', points), ('coupled_projected_disjoint.csv', disjoint)]:
        _write_csv(output/name, rows, writer.fieldnames)
    return dict(domain_rows=count, original_independent_blocks=32, independent_circuit_instances=1,
        frozen_definition_status_counts=dict(definitions), coupled_current_empirical_status_counts=dict(current),
        endpoint_transitions={key: [dict(native=a, coupled=b, count=n) for (a,b),n in sorted(value.items())]
                              for key, value in transitions.items()},
        saved_result_counts={prefix: {key: dict(value) for key, value in groups.items()} for prefix, groups in counts.items()},
        family_counts={key: dict(value) for key, value in sorted(families.items())},
        coupled_projected_exceedance_count=len(adverse), coupled_point_outside_count=len(points), coupled_projected_disjoint_count=len(disjoint),
        coupled_point_outside_rows=points, coupled_projected_disjoint_rows=disjoint,
        coupled_cdf_exceedances_matching_saved_tail_value=sum(row['coupled_cdf_projected_absolute_bound_matches_saved_tail_value'] for row in adverse))


def validate_parent_counts(result, reports):
    for prefix in ('native', 'coupled'):
        counts = result['saved_result_counts'][prefix]; report = reports[prefix]
        require(counts['planning_status'] == report['planning_status_counts'], 'SAVED_PLANNING_COUNTS_CHANGED:'+prefix)
        for key, reference in [('calibration_relation', 'calibration_interval_relations'), ('projected_relation', 'planning_interval_relations')]:
            observed = {k: v for k, v in counts[key].items() if k not in ('SOURCE_IDENTITY', 'UNDEFINED')}
            require(observed == report[reference], 'SAVED_INTERVAL_RELATION_COUNTS_CHANGED:'+prefix)
        require(counts['point_outside'].get('True', 0) == report['calibration_points_outside_budget'],
                'SAVED_POINT_OUTSIDE_COUNTS_CHANGED:'+prefix)


def run(config_path, project_root, output):
    settings = yaml.safe_load(config_path.read_text())
    require(settings.get('schema') == SCHEMA and settings.get('scope') == 'saved_calibration_result_classification'
            and settings.get('confirmation_execution_released') is False and settings.get('coverage_qualified') is False,
            'SAVED_UNQUALIFIED_RESULTS_ONLY_REQUIRED')
    require(settings['expected_endpoint_rows'] == 84330 and settings['planning_blocks'] == 128 and set(settings['inputs']) == set(FILES),
            'FULL_FROZEN_DOMAIN_AND_PLANNING_DESIGN_REQUIRED')
    reports, roots, paths, bindings = {}, {}, {}, []
    for key, ref in settings['inputs'].items():
        root = project_root/ref['path']; roots[key] = root
        require(sha256_file(root/'report.json') == ref['report_sha256'], 'BOUND_PARENT_REPORT_CHANGED:'+key)
        reports[key] = json.loads((root/'report.json').read_text())
        for name in FILES[key]:
            path = root/name; binding = reports[key]['outputs'][name]
            require(path.stat().st_size == binding['bytes'] and sha256_file(path) == binding['sha256'], 'BOUND_INPUT_CHANGED:'+name)
            paths[key, name] = path; bindings.append(dict(input=key, path=name, **binding))
    require(reports['coupled']['status'] == 'COMPLETE_COUPLED_CALIBRATION_JOINT_UNCERTAINTY_DIAGNOSTIC'
            and all(reports['coupled'][key] is False for key in FALSE_FLAGS)
            and reports['native']['status'] == 'COMPLETE_FROZEN_BUDGET_CAUSE_DIAGNOSIS'
            and reports['native_bands']['status'] == 'COMPLETE_CALIBRATION_JOINT_TAIL_UNCERTAINTY_DIAGNOSTIC'
            and reports['budget']['numerical_margins_frozen'] is True,
            'COMPLETE_BOUND_CALIBRATION_DIAGNOSTICS_REQUIRED')
    require(reports['native']['settings']['inputs']['uncertainty'] == settings['inputs']['native_bands']
            and reports['native']['frozen_budget_report_sha256'] == settings['inputs']['budget']['report_sha256']
            and reports['coupled']['settings']['inputs']['budget'] == settings['inputs']['budget']
            and reports['coupled']['frozen_margin_sha256'] == reports['native']['frozen_margin_sha256']
                == reports['budget']['outputs']['frozen_numerical_margins.csv']['sha256'], 'SHARED_FROZEN_BUDGET_PROVENANCE_REQUIRED')
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    ordered = [('budget', 'frozen_numerical_margins.csv'), ('native', 'calibration_budget_diagnosis.csv'),
               ('native_bands', 'endpoint_bands.csv'), ('coupled', 'calibration_budget_assessment.csv'), ('coupled', 'endpoint_bands.csv')]
    with ExitStack() as stack:
        tables = [csv.DictReader(stack.enter_context(paths[key].open(newline='', encoding='utf-8'))) for key in ordered]
        result = summarize_tables(tables, settings['expected_endpoint_rows'],
                                  reports['coupled']['joint_tail_probability_upper_zero_departure_planning'], output)
    validate_parent_counts(result, reports)
    result.update(status='COMPLETE_SAVED_COUPLED_BUDGET_GAP_DIAGNOSIS', settings=settings, settings_sha256=sha256_file(config_path),
        input_files=bindings, implementation_sha256=sha256_file(Path(__file__)),
        source_simulations=0, bootstrap_recomputed=False, margins_recomputed=False, planning_blocks=128,
        production_episodes=0, method_adopted_for_confirmation=False, coverage_qualified=False,
        uncertainty_coverage_qualified=False, scientific_fidelity_qualified=False, confirmation_execution_released=False,
        signed_gap_definition='Positive-side gap=upper-minus-margin; negative-side gap=minus-margin-minus-lower. Negative values are slack. Point uses estimate for both bounds.',
        tail_marker_interpretation='Saved projected CDF absolute bound numerically equals the saved zero-departure tail value at the original tolerance. This is descriptive equality, not proof that tail uncertainty caused the endpoint gap.',
        interpretation='Exact classification of saved approximate calibration diagnostics. A point outside budget is empirical displacement; a projected interval outside budget is a planning gap. Neither supports an independently confirmed numerical-error, fidelity or coverage claim.',
        wall_seconds=time.monotonic()-started,
        outputs={path.name: dict(bytes=path.stat().st_size, sha256=sha256_file(path)) for path in output.iterdir() if path.is_file()})
    freeze_json(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = run(args.config, args.project_root, args.output)
    print(json.dumps({k: result[k] for k in ['status', 'domain_rows', 'coupled_projected_exceedance_count', 'coupled_point_outside_count', 'coupled_projected_disjoint_count']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

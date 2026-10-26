"""Inspect saved penalty curves and optional null-reference selection without refitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(source, output):
    delivery = json.loads((source / 'delivery.json').read_text())
    manifest_path = source / 'payload_manifest.json'
    if sha(manifest_path) != delivery['payload_manifest_sha256']:
        raise ValueError('source payload manifest changed')
    files = {row['path']: row for row in json.loads(manifest_path.read_text())['files']}
    bindings = {}

    def bound(name):
        path = source / name
        digest = sha(path)
        if digest != files[name]['sha256'] or path.stat().st_size != files[name]['bytes']:
            raise ValueError('source artifact changed: ' + name)
        bindings[name] = digest
        return path

    analysis = json.loads(bound('analysis/analysis.json').read_text())
    features_path = bound('features/features.json')
    if sha(features_path) != analysis['feature_report_sha256']:
        raise ValueError('analysis and feature metadata differ')
    config = json.loads(features_path.read_text())['config']
    blocks = config['fit_blocks']
    models = analysis['models']
    targets = [(family, horizon) for family in config['future_statistics'] for horizon in config['horizons_ms']]
    if set(blocks) & set(config['selection_blocks']) or analysis['fitting_blocks'] != blocks:
        raise ValueError('fitting and selection block identities differ')
    if delivery['status'] != 'ARCHIVED_INDEPENDENTLY_REPLAYED_DEVELOPMENT_PROBE':
        raise ValueError('independently replayed original comparison required')
    selected = {}
    with bound('analysis/penalty_selection.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['model'], row['family'], int(row['horizon_ms']))
            if key in selected:
                raise ValueError('duplicate selected penalty')
            selected[key] = row
    curves = {}
    row_count = 0
    with bound('analysis/inner_validation.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['model'], row['family'], int(row['horizon_ms']), float(row['alpha']))
            block, mse = int(row['fit_validation_block']), float(row['mse'])
            if block not in blocks or not math.isfinite(mse) or mse < 0 or block in curves.setdefault(key, {}):
                raise ValueError('invalid or duplicate fitting-block CV row')
            curves[key][block] = mse
            row_count += 1
    expected_choices = {(model, family, horizon) for model in models for family, horizon in targets}
    expected_curves = {(*key, alpha) for key in expected_choices
                       for alpha in ([1.] if key[0] == 'cells' else config['regularization'])}
    if set(selected) != expected_choices or set(curves) != expected_curves:
        raise ValueError('complete saved model, target and penalty denominator required')
    rows = []
    for key in sorted(expected_choices):
        grid = [1.] if key[0] == 'cells' else config['regularization']
        means, curve_rows = [], []
        for alpha in grid:
            values = curves[(*key, alpha)]
            if set(values) != set(blocks):
                raise ValueError('all fitting folds required')
            mean = statistics.fmean(values[block] for block in blocks)
            means.append(mean)
            curve_rows.append(dict(alpha=alpha, mean_fit_block_mse=mean,
                                   fit_block_mse=[values[block] for block in blocks]))
        chosen = min(range(len(grid)), key=lambda index: means[index])
        saved = selected[key]
        if float(saved['alpha']) != grid[chosen] or not math.isclose(
                float(saved['fitting_cv_mse']), means[chosen], rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError('saved penalty is not the predeclared CV/grid-order minimum')
        location = ('baseline' if len(grid) == 1 else 'lower' if chosen == 0
                    else 'upper' if chosen == len(grid)-1 else 'interior')
        rows.append(dict(model=key[0], family=key[1], horizon_ms=key[2], selected_alpha=grid[chosen],
            selected_fit_mse=means[chosen], grid_location=location,
            active_features=int(saved['active_features']), effective_df=float(saved['effective_df']),
            strictly_decreasing_fit_mse=all(a > b for a, b in zip(means, means[1:])) if len(grid)>1 else None,
            last_step_mse_reduction=means[-2]-means[-1] if len(grid)>1 else None, curve=curve_rows))
    summary = {model: dict(Counter(row['grid_location'] for row in rows if row['model'] == model))
               for model in models}
    report = dict(status='COMPLETE_SAVED_FIT_CV_PENALTY_DIAGNOSTIC',
        recorded_utc=datetime.now(timezone.utc).isoformat(), input_sha256=bindings,
        source_delivery_sha256=sha(source/'delivery.json'), source_payload_manifest_sha256=sha(manifest_path),
        implementation_sha256=sha(Path(__file__)), fitting_blocks=blocks, saved_cv_rows=row_count,
        saved_penalty_choices_verified=len(rows), original_regularization_grid=config['regularization'],
        boundary_counts=summary, target_curves=rows, refits=0, neural_training_runs=0,
        raw_arrays_read=0, selection_outcomes_read=False, calibration_outcomes_accessed=False,
        test_outcomes_accessed=False, original_result_changed=False, scientific_qualification=False,
        interpretation='Search-boundary evidence concerns only the saved fitting-block grid. '
        'A boundary minimum cannot establish the optimum outside the grid, improvement on selection '
        'data, absence of historical information, closure, or scientific fidelity. No new penalty '
        'or model is selected here; the original finite-grid result and its failures remain intact.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {key: report[key] for key in ('status', 'saved_cv_rows', 'saved_penalty_choices_verified', 'boundary_counts')}


def null_reference(source, prior, output):
    """Read existing development errors only after fit-CV choices are written."""
    prior_delivery = json.loads((prior/'delivery.json').read_text())
    prior_file = next(row for row in prior_delivery['files'] if row['path'] == 'report.json')
    prior_path = prior/'report.json'
    if prior_delivery['status'] != 'COMPLETE' or prior_delivery['exit_code'] != 0:
        raise ValueError('completed saved-CV diagnostic required')
    if sha(prior_path) != prior_file['sha256'] or prior_path.stat().st_size != prior_file['bytes']:
        raise ValueError('saved-CV report changed')
    previous = json.loads(prior_path.read_text())
    if (previous['status'] != 'COMPLETE_SAVED_FIT_CV_PENALTY_DIAGNOSTIC'
            or sha(source/'delivery.json') != previous['source_delivery_sha256']
            or sha(source/'payload_manifest.json') != previous['source_payload_manifest_sha256']):
        raise ValueError('original source and saved-CV diagnosis must agree')
    files = {row['path']: row for row in json.loads((source/'payload_manifest.json').read_text())['files']}
    bindings = {}

    def bound(name):
        path = source/name
        digest = sha(path)
        if digest != files[name]['sha256'] or path.stat().st_size != files[name]['bytes']:
            raise ValueError('original result changed: ' + name)
        bindings[name] = digest
        return path

    config = json.loads(bound('features/features.json').read_text())['config']
    targets = [(family, horizon) for family in config['future_statistics'] for horizon in config['horizons_ms']]
    models = [representation+'_'+kind for representation in config['representations'] for kind in config['predictors']]
    cv = {(row['model'], row['family'], row['horizon_ms']): row for row in previous['target_curves']}
    expected = {(model, family, horizon) for model in ['cells', *models] for family, horizon in targets}
    if len(previous['target_curves']) != len(cv) or set(cv) != expected:
        raise ValueError('all original models and targets required')
    if previous['fitting_blocks'] != config['fit_blocks'] or set(config['fit_blocks']) & set(config['selection_blocks']):
        raise ValueError('fit and selection blocks must remain disjoint')
    choices, selection = [], {}
    for model in models:
        for family, horizon in targets:
            state = cv[model, family, horizon]
            cells = cv['cells', family, horizon]
            take = 'cells' if cells['selected_fit_mse'] <= state['selected_fit_mse'] else model
            selection[model, family, horizon] = take
            choices.append(dict(model=model, family=family, horizon_ms=horizon,
                original_alpha=state['selected_alpha'], state_fit_cv_mse=state['selected_fit_mse'],
                cells_fit_cv_mse=cells['selected_fit_mse'], chosen_existing_model=take,
                tie_rule='cells_on_exact_tie', used_development_outcome_for_choice=False))
    output.mkdir(parents=True, exist_ok=False)

    def write_table(name, rows):
        with (output/name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # The choices consume only the independently replayed fitting-block CV results.
    write_table('fit_cv_choices.csv', choices)
    choices_sha = sha(output/'fit_cv_choices.csv')
    errors = {}
    with bound('analysis/errors_by_block.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['model'], row['family'], int(row['horizon_ms']), int(row['block']))
            mse = float(row['mse'])
            if (key in errors or not math.isfinite(mse) or mse < 0
                    or int(row['rows']) != len(config['templates'])*len(config['origins_ms'])):
                raise ValueError('invalid original per-block error')
            errors[key] = mse
    if set(errors) != {(*key, block) for key in expected for block in config['selection_blocks']}:
        raise ValueError('complete original per-block denominator required')
    outcomes = []
    for model, family, horizon in selection:
        take = selection[model, family, horizon]
        for block in config['selection_blocks']:
            original = errors[model, family, horizon, block]
            chosen = errors[take, family, horizon, block]
            outcomes.append(dict(model=model, family=family, horizon_ms=horizon, block=block,
                chosen_existing_model=take, original_mse=original, selected_mse=chosen,
                cells_mse=errors['cells', family, horizon, block], selected_minus_original_mse=chosen-original))
    original_effects = {}
    with bound('analysis/history_effects.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['predictor'], row['family'], int(row['horizon_ms'])
            if key in original_effects:
                raise ValueError('duplicate original history effect')
            original_effects[key] = row
    if set(original_effects) != {(kind, family, horizon) for kind in config['predictors'] for family, horizon in targets}:
        raise ValueError('all original history contrasts required')
    effects, contrasts = [], []
    for kind in config['predictors']:
        for family, horizon in targets:
            recent, history = 'recent_'+kind, 'recent_and_history_'+kind
            a = [errors[recent, family, horizon, block] for block in config['selection_blocks']]
            b = [errors[history, family, horizon, block] for block in config['selection_blocks']]
            original_delta = [right-left for left, right in zip(a, b)]
            original = original_effects[kind, family, horizon]
            if (not math.isclose(statistics.fmean(original_delta), float(original['history_minus_recent_mse']),
                                 rel_tol=1e-10, abs_tol=1e-9)
                    or sum(value < 0 for value in original_delta) != int(original['blocks_improved'])
                    or int(original['blocks']) != len(a)):
                raise ValueError('original history contrast does not match all saved blocks')
            recent_take, history_take = selection[recent, family, horizon], selection[history, family, horizon]
            left = [errors[recent_take, family, horizon, block] for block in config['selection_blocks']]
            right = [errors[history_take, family, horizon, block] for block in config['selection_blocks']]
            delta = [r-l for l, r in zip(left, right)]
            for index, block in enumerate(config['selection_blocks']):
                contrasts.append(dict(predictor=kind, family=family, horizon_ms=horizon, block=block,
                    recent_selected_model=recent_take, history_selected_model=history_take,
                    recent_selected_mse=left[index], history_selected_mse=right[index],
                    selected_history_minus_recent_mse=delta[index],
                    original_history_minus_recent_mse=original_delta[index]))
            mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
            effects.append(dict(predictor=kind, family=family, horizon_ms=horizon,
                recent_selected_model=recent_take, history_selected_model=history_take,
                original_history_minus_recent_mse=statistics.fmean(original_delta),
                selected_history_minus_recent_mse=statistics.fmean(delta),
                selected_difference_block_sd=statistics.stdev(delta),
                selected_relative_mse_change=mean_right/mean_left-1 if mean_left else None,
                selected_blocks_improved=sum(value < 0 for value in delta),
                selected_blocks_equal=sum(value == 0 for value in delta), blocks=len(delta)))
    if sha(output/'fit_cv_choices.csv') != choices_sha:
        raise ValueError('choices changed after outcome access')
    write_table('development_errors_by_block.csv', outcomes)
    write_table('history_contrasts_by_block.csv', contrasts)
    write_table('history_contrast_summary.csv', effects)
    names = ['fit_cv_choices.csv', 'development_errors_by_block.csv', 'history_contrasts_by_block.csv', 'history_contrast_summary.csv']
    null_counts = {model: sum(row['model'] == model and row['chosen_existing_model'] == 'cells' for row in choices)
                   for model in models}
    report = dict(status='COMPLETE_EXISTING_NULL_REFERENCE_SENSITIVITY',
        recorded_utc=datetime.now(timezone.utc).isoformat(), diagnostic_question=
        'Does permitting the already fitted condition-only reference alter the fixed-grid history comparison?',
        prior_cv_diagnostic_sha256=sha(prior_path), source_delivery_sha256=sha(source/'delivery.json'),
        input_sha256=bindings, fitting_blocks=config['fit_blocks'], development_blocks=config['selection_blocks'],
        choices_written_before_development_errors_read=True, choices_sha256=choices_sha,
        decision_rule='lower_saved_fit_block_CV_MSE; exact_tie_cells',
        existing_candidates_only=True, selected_null_counts=null_counts,
        target_choices=len(choices), original_block_errors_read=len(errors), selected_block_errors=len(outcomes),
        original_history_effects_replayed=len(effects), contrast_rows=len(contrasts), history_effects=effects,
        refits=0, new_predictions=0, neural_training_runs=0, new_seed_streams=0,
        raw_arrays_read=0, development_selection_outcomes_read=True, calibration_outcomes_accessed=False,
        test_outcomes_accessed=False, original_result_changed=False, formal_model_selected=False,
        scientific_qualification=False, confidence_intervals=False, p_values=False,
        interpretation='Post-hoc diagnostic on previously exposed development blocks, not independent validation. '
        'Choices use the existing fitting-block CV results only. Including the existing condition-mean reference '
        'does not explore stronger finite regularization, other bandwidths, richer representations, or true '
        'state sufficiency. Neither favorable subsets nor null selections establish closure or production readiness.',
        implementation_sha256=sha(Path(__file__)),
        outputs={name:dict(bytes=(output/name).stat().st_size, sha256=sha(output/name)) for name in names})
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return {key:report[key] for key in ('status', 'selected_null_counts', 'target_choices',
        'selected_block_errors', 'original_history_effects_replayed', 'history_effects')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('penalty-grid', 'null-reference'), default='penalty-grid')
    parser.add_argument('--prior-diagnostic', type=Path)
    args = parser.parse_args()
    if args.mode == 'null-reference':
        if args.prior_diagnostic is None:
            parser.error('--prior-diagnostic required for null-reference mode')
        result = null_reference(args.source, args.prior_diagnostic, args.output)
    else:
        result = diagnose(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

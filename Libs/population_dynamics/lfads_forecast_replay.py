"""Independent all-unit and field-score replay of matched LFADS forecasts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .lfads import LFADSRandomStreams
from .lfads_trajectory_evaluation import (
    MATCHED_PROTOCOL, LEGACY_TRAINER_SHA256, BUDGET_TRAINER_SHA256,
    _loaded_model, _native_episode, _sample_payload, _verified_training,
)
from .lfads_trajectory_training import _checkpoint_budget, _write_json
from .poisson_trajectory_evaluation import fitting_scale
from .population_forecast_replay import (
    METRIC_ATOL, METRIC_RTOL, _exact_array, _metric_comparisons, _project_units,
    _score_rows, numpy_unit_metrics,
)
from .provenance import sha256_file
from .trajectory_forecast_analysis import pair_sets, replay_scores
from .trajectory_sample_metrics import unit_count_metrics
from .lfads_evaluator_compatibility import verify_legacy_evaluator


LEGACY_UNIT_METRIC_SHA256 = '9a7ca01e10fe5e9424cc594cbfef20c3f287ac9f684a07a9abd99fd2e6c0c02a'


def _same_execution_migration(saved, current, run, precision):
    if saved == current:
        return True
    # Original20-epoch forecasts predate the explicit trainer metadata bridge.
    # _verified_training already verifies the unchanged numerical adapter and
    # exact legacy parent. Admit precisely that bridge during explicit rescoring.
    expected = dict(schema='lfads_pinned_budget_trainer_metadata_bridge_v1',
        legacy_trainer_sha256=LEGACY_TRAINER_SHA256,budget_trainer_sha256=BUDGET_TRAINER_SHA256,
        saved_trainer_sha256=LEGACY_TRAINER_SHA256,inference_trainer_sha256=BUDGET_TRAINER_SHA256,
        receipt_trainer_sha256=LEGACY_TRAINER_SHA256,checkpoint_scope='legacy_parent',
        verified_continuation_sha256=None,
        scope='trainer metadata only; numerical adapter migration evidence remains unchanged')
    if (precision is None or not isinstance(saved,dict) or not isinstance(current,dict)
            or run.get('continuation') is not None
            or run['binding']['sources']['lfads_trajectory_training.py'] != LEGACY_TRAINER_SHA256
            or 'trainer_bridge' in saved or current.get('trainer_bridge') != expected
            or {k:v for k,v in current.items() if k != 'trainer_bridge'} != saved):
        return False
    precision['verified_legacy_parent_trainer_bridge'] = expected
    return True


def _precision_binding(evaluation, qualification):
    """Admit only the diagnosed arithmetic repair; never broaden tolerances."""
    qualification = Path(qualification)
    proof = json.loads(qualification.read_text())
    current = sha256_file(Path(__file__).with_name('trajectory_sample_metrics.py'))
    if (evaluation['unit_metrics_implementation_sha256'] != LEGACY_UNIT_METRIC_SHA256
            or proof['status'] != 'REPRODUCED_FP32_CANCELLATION_EXACT_CORRECTION_PASS'
            or proof['candidate_metric_sha256'] != current
            or proof['original_unit_metrics_sha256'] != LEGACY_UNIT_METRIC_SHA256
            or proof['all_saved_fields_and_latent_paths_exact'] is not True
            or not proof['comparisons'] or not all(r['revised_passed'] for r in proof['comparisons'])
            or proof['counts_sha256'] != sha256_file(qualification.parent/'failed_case_counts.npz')
            or proof['scientific_claim_qualified'] is not False):
        raise ValueError('verified original-case exact integer correction required before rescoring')
    return dict(schema='lfads_exact_count_correction_rescore_v1',
                legacy_unit_metrics_sha256=LEGACY_UNIT_METRIC_SHA256,
                corrected_unit_metrics_sha256=current, qualification_sha256=sha256_file(qualification),
                unchanged_metric_tolerance=dict(rtol=METRIC_RTOL,atol=METRIC_ATOL),
                unchanged_output_dtype='float32', unchanged_estimand=True,
                original_forecast_evaluation_sha256=None)


def replay(acquisition, fields, training, forecasts, output, device='cuda', *, inference_migration=None,
           rescore_output=None, precision_qualification=None, legacy_evaluator_source=None, prior_refit=None, conditional_prior=None):
    acquisition, fields, training, forecasts, output = map(Path, (acquisition, fields, training, forecasts, output))
    csv_path = output.with_suffix('.csv')
    if output.suffix != '.json' or output.exists() or csv_path.exists():
        raise ValueError('new JSON replay output and CSV sibling required')
    if (rescore_output is None) != (precision_qualification is None):
        raise ValueError('rescore output and actual numerical qualification must be supplied together')
    started = time.perf_counter()
    evaluation = json.loads((forecasts / 'evaluation.json').read_text())
    if (evaluation['status'] != 'EVALUATED_DEVELOPMENT' or evaluation['model_type'] != 'lfads'
            or evaluation['configuration'] != MATCHED_PROTOCOL):
        raise ValueError('completed matched LFADS development evaluation required')
    precision = None if rescore_output is None else _precision_binding(evaluation, precision_qualification)
    cohort, run, directory, selected, migration = _verified_training(acquisition, fields, training, inference_migration)
    prior_parameters = None
    if prior_refit is not None:
        from .lfads_prior_refit import load_prior_parameters
        prior_parameters, prior_binding = load_prior_parameters(prior_refit, run=run, selected=selected,
            cohort_binding=cohort.binding, training_run_sha256=sha256_file(training / 'run.json'))
        if evaluation.get('prior_refit') != prior_binding:
            raise ValueError('forecast prior-refit identity changed')
    elif 'prior_refit' in evaluation:
        raise ValueError('forecast replay requires its fitted AR prior')
    mean_prior = None
    if conditional_prior is not None:
        if prior_refit is None:
            raise ValueError('conditional mean replay requires its fitted AR prior')
        from .lfads_conditional_prior import load_mean_prior
        mean_prior, mean_binding = load_mean_prior(conditional_prior, prior_binding,
            training_seed=run['binding']['seed'],
            network_budget_epochs=_checkpoint_budget(run, run['binding']['config'])[0],
            selected_model_artifacts=selected['model_artifacts'])
        if evaluation.get('conditional_prior') != mean_binding:
            raise ValueError('forecast conditional-prior identity changed')
    elif 'conditional_prior' in evaluation:
        raise ValueError('forecast replay requires its conditional mean prior')
    binding = run['binding']
    if (evaluation['training_binding'] != binding or evaluation['training_run_sha256'] != sha256_file(training / 'run.json')
            or evaluation['selected_training_sha256'] != sha256_file(directory / 'training.json')
            or evaluation['selected_model_artifacts'] != selected['model_artifacts']
            or not _same_execution_migration(evaluation['inference_migration'],migration,run,precision)):
        raise ValueError('evaluated training or inference-migration identity changed')
    evaluator_compatibility = None
    for key, name in [('evaluator_implementation_sha256', 'lfads_trajectory_evaluation.py'),
                      ('unit_metrics_implementation_sha256', 'trajectory_sample_metrics.py')]:
        if key == 'unit_metrics_implementation_sha256' and precision is not None:
            continue  # The explicit numerical binding above verifies old and new implementations.
        if evaluation[key] != sha256_file(Path(__file__).with_name(name)):
            if (key != 'evaluator_implementation_sha256' or legacy_evaluator_source is None
                    or prior_refit is not None):
                raise ValueError('recorded evaluation implementation changed')
            evaluator_compatibility = verify_legacy_evaluator(
                evaluation[key],legacy_evaluator_source,Path(__file__).with_name(name))
            if precision is not None:
                precision['verified_legacy_evaluator'] = evaluator_compatibility
    for name, digest in evaluation['artifacts'].items():
        if Path(name).name != name or sha256_file(forecasts / name) != digest:
            raise ValueError('forecast artifact changed')
    if sha256_file(forecasts / 'scores.csv') != evaluation['scores_sha256']:
        raise ValueError('forecast metric table changed')
    # Shared row parser predates the explicit fitting-scale artifact; verify it
    # below and pass precisely its expected ensemble manifest to that parser.
    row_record = dict(evaluation, artifacts={k:v for k,v in evaluation['artifacts'].items()
                                           if k != 'fitting_field_scale.npz'})
    indexed = _score_rows(forecasts, row_record, cohort.validation)
    if precision is not None:
        rescore_output = Path(rescore_output)
        rescore_output.mkdir(parents=True, exist_ok=False)
        precision['original_forecast_evaluation_sha256'] = sha256_file(forecasts/'evaluation.json')
        precision['original_scores_sha256'] = sha256_file(forecasts/'scores.csv')
        # Keep each original forecast immutable. The same samples are regenerated
        # below and checked against all saved fields/paths before new scores commit.
        for name in evaluation['artifacts']:
            shutil.copy2(forecasts/name,rescore_output/name)
            if sha256_file(rescore_output/name) != evaluation['artifacts'][name]:
                raise ValueError('copied rescore input differs: '+name)
        torch.set_num_threads(binding['config']['cpu_threads'])
    mean, sd, metadata = fitting_scale(fields, cohort.fit)
    with np.load(forecasts / 'fitting_field_scale.npz', allow_pickle=False) as scale:
        for name, value in [('mean_hz', mean), ('standard_deviation_hz', sd), *metadata.items()]:
            _exact_array(scale[name], value, 'fitting scale or field metadata changed: ' + name)
    pairs = pair_sets(cohort.field_ids)
    config = evaluation['configuration']
    bin_ms = binding['config']['bin_ms']
    horizons = [int(h / bin_ms) for h in config['horizons_ms']]
    longest, draws = max(horizons), config['joint_samples']
    native = _native_episode(cohort, cohort.validation[0])
    intervals, source = native.minimum_interval_ticks.copy(), native.source_identity
    if source != evaluation['native_source_identity']:
        raise ValueError('evaluated native source changed')
    del native
    comparisons, origins, field_deltas = [], [], []
    result = dict(status='RUNNING', training_seed=binding['seed'], model_type='lfads',
        evaluation_sha256=sha256_file(forecasts / 'evaluation.json'),
        dataset_sha256=binding['dataset_sha256'], scores_sha256=evaluation['scores_sha256'],
        metric_tolerance=dict(rtol=METRIC_RTOL, atol=METRIC_ATOL),
        field_tolerance=dict(rtol=1e-10, atol=1e-12), scientific_fidelity_qualified=False,
        replay_implementation_sha256=sha256_file(Path(__file__)))
    if evaluator_compatibility is not None:
        result['verified_legacy_evaluator'] = evaluator_compatibility
    if prior_refit is not None:
        result['prior_refit'] = prior_binding
    if mean_prior is not None:
        result['conditional_prior'] = mean_binding
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    try:
        for origin_index, origin in enumerate(config['origins_ms']):
            boundary = int(origin / bin_ms)
            model = _loaded_model(directory, cohort, binding, boundary, longest, device,
                prior_parameters=prior_parameters)
            try:
                for case_index, case in enumerate(cohort.validation):
                    data = _native_episode(cohort, case, source, intervals)
                    count, stimulus = data.window(0, data.duration_ticks)
                    state = model.infer_state(dict(counts=count[None, :boundary], stimulus=stimulus[None, :boundary]),
                        metadata=dict(unit_ids=list(map(int, cohort.unit_ids)), bin_seconds=bin_ms / 1000))
                    seed = config['sampling_seed_base'] + binding['seed'] * 100000 + case_index * 100 + origin_index * 3
                    streams = LFADSRandomStreams.from_seeds(seed, seed + 1, seed + 2)
                    future_stimulus = stimulus[None, boundary:boundary + longest]
                    if mean_prior is None:
                        sample = model.simulate(state, future_stimulus, streams, draws)
                    else:
                        from .lfads_conditional_prior import simulate_with_mean
                        sample = simulate_with_mean(model, state, future_stimulus, streams, draws,
                            mean_prior.curve(case['template_id'], boundary, longest))
                    paths = _sample_payload(sample, cohort.unit_ids, draws, boundary, longest, bin_ms / 1000, binding['config']['model'])
                    if mean_prior is not None:
                        paths['lfads_base_future_controller_inputs'] = sample.base_future_controller_inputs[:, 0]
                    generated, observed = sample.counts[:, 0], count[boundary:boundary + longest]
                    del sample
                    fields_pred, fields_true = _project_units(generated, observed, cohort.unit_field_index, len(cohort.field_ids))
                    name = f"{case['case_id']}_origin_{origin:g}.npz"
                    with np.load(forecasts / name, allow_pickle=False) as saved:
                        if set(saved.files) != {'sampled_field_counts', 'observed_field_counts', *paths}:
                            raise ValueError('LFADS saved field/path schema changed')
                        for key, value in [('sampled_field_counts', fields_pred), ('observed_field_counts', fields_true), *paths.items()]:
                            _exact_array(saved[key], value, 'exact LFADS path/field replay mismatch: ' + key)
                    first = case_index == origin_index == 0
                    if first:
                        with np.load(forecasts / 'full_unit_samples.npz', allow_pickle=False) as saved:
                            for key, value in [('sampled_unit_counts', generated), ('unit_ids', cohort.unit_ids),
                                               ('initial_ready_in_ticks', data.ready_at(boundary * data.bin_ticks)), *paths.items()]:
                                _exact_array(saved[key], value, 'exact saved LFADS full-unit replay mismatch: ' + key)
                    metrics, exact, consumed = numpy_unit_metrics(generated, observed, intervals,
                        data.bin_ticks, horizons, config['refractory_support_windows_bins'])
                    scaled = (fields_pred / (metadata['unit_count'] * bin_ms / 1000) - mean) / sd
                    truth = (fields_true / (metadata['unit_count'] * bin_ms / 1000) - mean) / sd
                    scope = 'exact_saved_full_unit_array' if first else 'regenerated_counts_no_saved_full_unit_array'
                    for ms, horizon in zip(config['horizons_ms'], horizons):
                        row = indexed[(case['case_id'], float(origin), float(ms))]
                        if precision is not None:
                            row.update(unit_count_metrics(torch.from_numpy(generated[:,:horizon]),
                                torch.from_numpy(observed[:horizon]).float(), intervals,
                                data.bin_ticks, config['refractory_support_windows_bins']))
                        comparisons.extend(_metric_comparisons(metrics[horizon], exact[horizon], row,
                            dict(initial_seed=seed, dynamics_seed=seed + 1, observation_seed=seed + 2), scope))
                        for key, value in replay_scores(scaled[:, :horizon], truth[:horizon], pairs, config['temporal_lags_bins']).items():
                            reported = float(row[key]);delta = abs(value - reported)
                            if not np.isfinite(value) or not np.isfinite(reported) or not np.isclose(value, reported, rtol=1e-10, atol=1e-12):
                                raise ValueError('independent field score mismatch: ' + key)
                            field_deltas.append(delta)
                    origins.append(dict(case_id=case['case_id'], origin_ms=origin,
                        sampled_unit_count_values_consumed=consumed, full_unit_reference=scope,
                        exact_field_and_lfads_paths=True))
                    result.update(verified_origins=len(origins), field_scores_recomputed=len(field_deltas))
                    _write_json(output, result)
                    print(json.dumps(dict(status='REPLAYED_LFADS_ORIGIN', seed=binding['seed'], case_id=case['case_id'],
                        origin_ms=origin, completed_origins=len(origins))), flush=True)
                    del generated, observed, paths, state, count, stimulus, data
            finally:
                model.close()
                model = None
        if precision is not None:
            with (rescore_output/'scores.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(next(iter(indexed.values()))))
                writer.writeheader();writer.writerows(indexed.values())
            rescored = dict(evaluation, scores_sha256=sha256_file(rescore_output/'scores.csv'),
                numerical_rescore=precision,
                inference_migration=migration,
                evaluator_implementation_sha256=sha256_file(Path(__file__).with_name('lfads_trajectory_evaluation.py')),
                unit_metrics_implementation_sha256=precision['corrected_unit_metrics_sha256'],
                unit_metric_estimator='float32 outputs; exact integer corrected-MSE numerator; other estimands unchanged',
                rescore_implementation_sha256=sha256_file(Path(__file__)),
                rescore_cost_scope='original forecast clocks retained; complete rescore and independent replay cost in replay receipt')
            _write_json(rescore_output/'evaluation.json',rescored)
            result.update(evaluation_sha256=sha256_file(rescore_output/'evaluation.json'),
                          scores_sha256=rescored['scores_sha256'],numerical_rescore=precision)
        passed = all(row['passed'] for row in comparisons)
        result.update(status='PASS' if passed else 'MISMATCH', verified_origins=len(origins),
            verified_score_rows=len(indexed), metric_comparisons=len(comparisons),
            field_scores_recomputed=len(field_deltas), maximum_field_absolute_difference=max(field_deltas),
            all_forecast_unit_metrics_replayed=passed, saved_full_unit_origins_compared=1,
            regenerated_origins_without_saved_full_unit_arrays=len(origins)-1,
            all_origin_unit_counts_compared_to_saved_full_arrays=False,
            sampled_unit_count_values_consumed=sum(row['sampled_unit_count_values_consumed'] for row in origins),
            origins=origins, elapsed_seconds=time.perf_counter()-started)
        with csv_path.open('x', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
        result.update(csv_artifact=csv_path.name, csv_sha256=sha256_file(csv_path))
        _write_json(output, result)
        if not passed:
            raise ValueError('frozen independent unit-metric tolerance failed')
        return result
    except BaseException as error:
        result.update(status='FAILED' if result['status']=='RUNNING' else result['status'], error=str(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
        _write_json(output, result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition','fields','training','forecasts','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--inference-migration', type=Path)
    parser.add_argument('--rescore-output', type=Path,
                        help='New forecast directory for the qualified exact-count arithmetic repair')
    parser.add_argument('--precision-qualification',type=Path,
                        help='Actual failed-case count/oracle proof required with --rescore-output')
    parser.add_argument('--legacy-evaluator-source',type=Path,
                        help='Pinned historical evaluator; numerical AST must match outside separately verified trainer metadata')
    parser.add_argument('--prior-refit', type=Path)
    parser.add_argument('--conditional-prior', type=Path)
    args=parser.parse_args()
    result=replay(args.acquisition,args.fields,args.training,args.forecasts,args.output,args.device,
                  inference_migration=args.inference_migration,rescore_output=args.rescore_output,
                  precision_qualification=args.precision_qualification,legacy_evaluator_source=args.legacy_evaluator_source,
                  prior_refit=args.prior_refit, conditional_prior=args.conditional_prior)
    print(json.dumps({k:v for k,v in result.items() if k!='origins'}))


if __name__=='__main__':
    main()

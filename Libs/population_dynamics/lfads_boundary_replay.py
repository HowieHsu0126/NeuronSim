"""Independent all-arm replay of completed LFADS boundary diagnostics.

Regeneration uses the same source-bound generator. Unit metrics and projection
are independently recomputed in NumPy, and field scores use the existing SciPy
replay implementation. Only the first case has archived full-unit arrays; every
case has exact field/latent replay. No diagnostic arm becomes a causal forecast.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from Libs.population_dynamics import lfads_boundary_diagnostic as diagnostic_module
from Libs.population_dynamics import trajectory_sample_metrics as unit_metrics_module
from Libs.population_dynamics.lfads import LFADSConditionalSimulator, SOURCE_REVISION
from Libs.population_dynamics.lfads_boundary_analysis import read_rows
from Libs.population_dynamics.lfads_boundary_diagnostic import (
    BRANCHES, PROTOCOL, LFADSBoundaryDiagnostic, branch_role,
)
from Libs.population_dynamics.lfads_trajectory_evaluation import (
    _native_episode, _sample_payload, _verified_training,
)
from Libs.population_dynamics.lfads_trajectory_training import _write_json
from Libs.population_dynamics.poisson_trajectory_evaluation import fitting_scale
from Libs.population_dynamics.population_forecast_replay import (
    METRIC_ATOL, METRIC_RTOL, UNIT_CHUNK_SIZE, _exact_array,
    _project_units, numpy_unit_metrics,
)
from Libs.population_dynamics.provenance import sha256_file
from Libs.population_dynamics.trajectory_forecast_analysis import pair_sets, replay_scores


FIELD_RTOL = 1e-10
FIELD_ATOL = 1e-12


def conditional_count_means(means, horizons, *, chunk_units=UNIT_CHUNK_SIZE):
    """Consume every conditional mean with float64 per-unit/time reductions."""
    values = np.asarray(means)
    if (values.ndim != 4 or values.shape[1] != 1 or min(values.shape) < 1
            or values.dtype.kind != 'f' or type(chunk_units) is not int or chunk_units < 1
            or not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or not 1 <= h <= values.shape[2] for h in horizons)
            or max(horizons) != values.shape[2]):
        raise ValueError('COMPLETE_IDENTIFIED_CONDITIONAL_COUNT_MEANS_REQUIRED')
    totals = {h: 0. for h in horizons}
    consumed = 0
    for start in range(0, values.shape[-1], chunk_units):
        part = values[:, 0, :, start:start + chunk_units].astype(np.float64)
        if not np.isfinite(part).all() or (part < 0).any():
            raise ValueError('FINITE_NONNEGATIVE_CONDITIONAL_COUNT_MEANS_REQUIRED')
        consumed += part.size
        for h in horizons:
            # Independent grouping from the evaluator's single flattened sum.
            per_unit = part[:, :h].sum(axis=1, dtype=np.float64)
            totals[h] += float(per_unit.sum(axis=1, dtype=np.float64).sum(dtype=np.float64))
    if consumed != values.size:
        raise ValueError('INCOMPLETE_CONDITIONAL_MEAN_DENOMINATOR')
    denominator = values.shape[0] * values.shape[-1]
    return {h: total / denominator for h, total in totals.items()}, consumed


def _float_comparison(actual, reference, *, rtol, atol, label):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('SOURCE_ENDPOINT_SHAPE_OR_DTYPE_MISMATCH:' + label)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('NONFINITE_SOURCE_ENDPOINT:' + label)
    delta = np.abs(actual.astype(np.float64) - reference)
    allowance = atol + rtol * np.abs(reference.astype(np.float64))
    result = dict(bitwise_equal=bool(np.array_equal(actual, reference)),
        within_tolerance=bool(np.all(delta <= allowance)),
        max_abs_error=float(delta.max()), maximum_tolerance_fraction=float((delta / allowance).max()))
    if not result['within_tolerance']:
        raise ValueError('RAW_RESPONSE_SOURCE_ENDPOINT_MISMATCH:' + label)
    return result


def raw_response_endpoint(model, prepared, counts, stimulus, sample, *, rtol, atol):
    """Native full-response batch, with only source Gaussian noises imposed.

    The source uses the actual joint-sample batch. It feeds neither cached encoder values
    nor posterior parameters, initial latent values, controller outputs or the
    generator boundary. All source encoders and feedback execute directly.
    """
    total, h = model.history_bins + model.horizon_bins, model.history_bins
    y = model._counts(counts, total)
    if y.shape[0] != 1:
        raise ValueError('RAW_RESPONSE_REFERENCE_REQUIRES_BATCH_ONE')
    u = model._stimulus(stimulus, 1, total)
    source = model._full.model
    draws = prepared.n_samples
    feed = source.build_feed_dict('recording', np.broadcast_to(y, (draws, total, len(model.unit_ids))),
                                  np.broadcast_to(u, (draws, total, model.input_size)), keep_prob=1.)
    feed[source.posterior_zs_g0.noise_bxn] = prepared.initial_standard_noise[:, 0]
    for tick in range(h + prepared.bins):
        feed[source.posterior_zs_co[tick].noise_bxn] = (
            prepared.history_standard_noise[:, tick] if tick < h
            else prepared.future_standard_noise[:, tick - h])
    reference = model._read_future(feed, draws, prepared.bins)
    arrays = {name: _float_comparison(getattr(sample, name), reference[name],
        rtol=rtol, atol=atol, label=name) for name in
        ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states')}
    return dict(reference='raw_complete_response_native_source', draw_indices=list(range(draws)),
        response_batch_size=draws, response_bins=total, uses_future_responses=True,
        imposed_values='IC and controller standardized Gaussian noises only',
        controller_or_boundary_imposed=False, encoders_cached=False,
        exact_posterior_claim=False, role='response_conditioned_diagnostic',
        rtol=rtol, atol=atol, arrays=arrays, all_within_tolerance=True)


def _scope_contract(evaluation, indexed):
    if evaluation['configuration'] != PROTOCOL:
        raise ValueError('EXACT_FROZEN_BOUNDARY_DIAGNOSTIC_PROTOCOL_REQUIRED')
    expected_roles = {name: branch_role(name) for name in BRANCHES}
    if evaluation.get('branch_roles') != expected_roles:
        raise ValueError('DIAGNOSTIC_BRANCH_ROLE_OR_FUTURE_RESPONSE_SCOPE_CHANGED')
    for key in ('formal_test_accessed', 'calibration_outcomes_accessed', 'scientific_claim_qualified',
                'inverse_intervention_evidence', 'mechanistic_microstate_oracle', 'weights_updated'):
        if evaluation.get(key) is not False:
            raise ValueError('DIAGNOSTIC_EVIDENCE_SCOPE_CHANGED:' + key)
    for row in indexed.values():
        if row.get('method') != 'lfads':
            raise ValueError('BOUNDARY_DIAGNOSTIC_METHOD_CHANGED')
        for key, expected in expected_roles[row['branch']].items():
            observed = row.get(key)
            if observed != (str(expected) if isinstance(expected, bool) else expected):
                raise ValueError('ROW_BRANCH_ROLE_OR_FUTURE_RESPONSE_SCOPE_CHANGED:' + key)


def _metric_rows(values, exact, recorded, *, family, scope, sampling_seeds):
    comparisons = []
    rtol, atol = (FIELD_RTOL, FIELD_ATOL) if family == 'field' else (METRIC_RTOL, METRIC_ATOL)
    for name, actual in values.items():
        reported = recorded[name]
        if actual is None or reported is None:
            if actual is not None or reported is not None:
                raise ValueError('METRIC_AVAILABILITY_MISMATCH:' + name)
            difference = allowance = None
            passed = True
        else:
            reported = float(reported)
            if not np.isfinite(actual) or not np.isfinite(reported):
                raise ValueError('NONFINITE_INDEPENDENT_REPLAY_METRIC:' + name)
            difference, allowance = abs(actual - reported), atol + rtol * abs(reported)
            passed = difference <= allowance
        numerator, denominator = exact.get(name, (None, None))
        comparisons.append(dict(training_seed=int(recorded['seed']), case_id=recorded['case_id'],
            block=int(recorded['block']), template_id=recorded['template_id'], branch=recorded['branch'],
            role=recorded['role'], origin_ms=float(recorded['origin_ms']), horizon_ms=float(recorded['horizon_ms']),
            initial_seed=sampling_seeds[0], dynamics_seed=sampling_seeds[1], observation_seed=sampling_seeds[2],
            full_unit_reference=scope, metric_family=family, metric=name,
            reported_value=reported, independent_value=actual, absolute_difference=difference,
            allowed_difference=allowance, rtol=rtol, atol=atol, passed=bool(passed),
            exact_numerator=numerator, exact_denominator=denominator))
    return comparisons


def _saved_branch(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError('SAVED_BOUNDARY_BRANCH_SCHEMA_CHANGED')
        for name, value in expected.items():
            _exact_array(saved[name], value, 'EXACT_BOUNDARY_BRANCH_REPLAY_MISMATCH:' + name)


def replay(acquisition, fields, training, diagnostics, output, device='cuda', *, inference_migration=None):
    acquisition, fields, training, diagnostics, output = map(Path,
        (acquisition, fields, training, diagnostics, output))
    csv_path = output.with_suffix('.csv')
    if output.suffix != '.json' or output.exists() or csv_path.exists():
        raise ValueError('NEW_JSON_REPLAY_OUTPUT_AND_CSV_SIBLING_REQUIRED')
    if device not in ('cpu', 'cuda'):
        raise ValueError('EXPLICIT_CPU_OR_CUDA_REQUIRED')
    if device == 'cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise ValueError('QUALIFIED_LFADS_CUDA_REQUIRES_NVIDIA_TF32_OVERRIDE_0')
    started = time.perf_counter()
    evaluation = json.loads((diagnostics / 'evaluation.json').read_text())
    indexed = read_rows(diagnostics, evaluation)
    _scope_contract(evaluation, indexed)
    cohort, run, directory, selected, migration = _verified_training(
        acquisition, fields, training, inference_migration)
    binding, cfg = run['binding'], evaluation['configuration']
    if (evaluation['training_binding'] != binding or evaluation['training_seed'] != binding['seed']
            or evaluation['training_run_sha256'] != sha256_file(training / 'run.json')
            or evaluation['selected_training_sha256'] != sha256_file(directory / 'training.json')
            or evaluation['selected_checkpoint'] != run['best_checkpoint']
            or evaluation['selected_epoch'] != run['best_epoch']
            or evaluation['selected_model_artifacts'] != selected['model_artifacts']
            or evaluation['source_revision'] != SOURCE_REVISION
            or evaluation['inference_migration'] != migration
            or any(evaluation[k] != binding[k] for k in ('dataset_sha256', 'acquisition_design_sha256'))):
        raise ValueError('EVALUATED_TRAINING_SOURCE_OR_MIGRATION_IDENTITY_CHANGED')
    if (len(cohort.unit_ids) != cfg['expected_units'] or len(cohort.validation) != cfg['expected_cases_per_seed']
            or binding['config']['history_bins'] != cfg['original_history_bins']
            or binding['config']['horizon_bins'] != cfg['original_horizon_bins']
            or binding['config']['bin_ms'] != cfg['bin_ms']):
        raise ValueError('SOURCE_BOUND_DIAGNOSTIC_DIMENSIONS_CHANGED')
    config_path = Path(diagnostic_module.__file__).resolve().parents[1] / 'configs/neural_population_lfads_boundary_diagnostic.yaml'
    for key, path in (
        ('configuration_sha256', config_path),
        ('evaluator_implementation_sha256', Path(diagnostic_module.__file__)),
        ('unit_metrics_implementation_sha256', Path(unit_metrics_module.__file__))):
        if evaluation[key] != sha256_file(path):
            raise ValueError('RECORDED_DIAGNOSTIC_IMPLEMENTATION_CHANGED:' + key)
    expected_artifacts = {'fitting_field_scale.npz',
        *(f"{case['case_id']}_{branch}.npz" for case in cohort.validation for branch in BRANCHES),
        *('full_unit_' + branch + '.npz' for branch in BRANCHES)}
    if set(evaluation['artifacts']) != expected_artifacts:
        raise ValueError('COMPLETE_REGISTERED_BOUNDARY_ARTIFACT_SET_REQUIRED')
    for name, digest in evaluation['artifacts'].items():
        if Path(name).name != name or sha256_file(diagnostics / name) != digest:
            raise ValueError('BOUNDARY_DIAGNOSTIC_ARTIFACT_CHANGED:' + name)
    case_bindings = evaluation['case_bindings']
    if ([row['case_id'] for row in case_bindings] != [case['case_id'] for case in cohort.validation]
            or len({row['case_id'] for row in case_bindings}) != cfg['expected_cases_per_seed']):
        raise ValueError('ORDERED_COMPLETE_SOURCE_CASE_BINDINGS_REQUIRED')
    mean, sd, metadata = fitting_scale(fields, cohort.fit)
    with np.load(diagnostics / 'fitting_field_scale.npz', allow_pickle=False) as scale:
        expected_scale = dict(mean_hz=mean, standard_deviation_hz=sd, **metadata)
        if set(scale.files) != set(expected_scale):
            raise ValueError('FITTING_SCALE_SCHEMA_CHANGED')
        for name, value in expected_scale.items():
            _exact_array(scale[name], value, 'FITTING_SCALE_OR_METADATA_CHANGED:' + name)
    pairs = pair_sets(cohort.field_ids)
    h, bins, draws = cfg['original_history_bins'], cfg['forecast_horizon_bins'], cfg['joint_samples']
    horizons = [int(ms / cfg['bin_ms']) for ms in cfg['horizons_ms']]
    native = _native_episode(cohort, cohort.validation[0])
    intervals, source = native.minimum_interval_ticks.copy(), native.source_identity
    if source != evaluation['native_source_identity']:
        raise ValueError('EVALUATED_NATIVE_SOURCE_CHANGED')
    del native
    result = dict(status='RUNNING', model_type='lfads_boundary_diagnostic', training_seed=binding['seed'],
        evaluation_sha256=sha256_file(diagnostics / 'evaluation.json'), scores_sha256=evaluation['scores_sha256'],
        dataset_sha256=binding['dataset_sha256'], selected_model_artifacts=selected['model_artifacts'],
        metric_tolerance=dict(rtol=METRIC_RTOL, atol=METRIC_ATOL),
        field_tolerance=dict(rtol=FIELD_RTOL, atol=FIELD_ATOL),
        source_endpoint_tolerance=dict(rtol=cfg['endpoint_rtol'], atol=cfg['endpoint_atol']),
        verified_cases=0, verified_branches=0, verified_score_rows=0,
        all_branch_unit_metrics_replayed=False, all_conditional_means_replayed=False,
        all_branch_full_unit_arrays_compared=False, scientific_claim_qualified=False,
        formal_test_accessed=False, inverse_intervention_evidence=False,
        branch_roles=evaluation['branch_roles'], branch_replays=[], endpoint_checks=[],
        raw_response_endpoint=None, replay_implementation_sha256=sha256_file(Path(__file__)),
        unit_replay_implementation_sha256=sha256_file(Path(numpy_unit_metrics.__code__.co_filename)),
        field_replay_implementation_sha256=sha256_file(Path(replay_scores.__code__.co_filename)),
        replay_scope='exact saved fields, latent paths and metadata for every branch; exact saved full-unit counts for first case only')
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    comparisons = []
    try:
        graph_started = time.perf_counter()
        with LFADSConditionalSimulator.load(directory, device=device) as model:
            cohort.verify_model(model)
            diagnostic = LFADSBoundaryDiagnostic(model)
            result.update(graph_setup_seconds=time.perf_counter() - graph_started,
                          execution_device=model.execution_device)
            for case_index, case in enumerate(cohort.validation):
                data = _native_episode(cohort, case, source, intervals)
                count, stimulus = data.window(0, data.duration_ticks)
                seed = cfg['sampling_seed_base'] + binding['seed'] * 100000 + case_index * 100
                seeds = (seed, seed + 1, seed + 2)
                case_record = case_bindings[case_index]
                if case_record['binding'] != data.binding or case_record['sampling_seeds'] != list(seeds):
                    raise ValueError('CASE_SOURCE_BINDING_OR_SAMPLING_SEEDS_CHANGED')
                prepared = diagnostic.prepare(count[None], stimulus[None], horizon_bins=bins,
                    n_samples=draws, sampling_seeds=seeds)
                observed = count[h:h + bins]
                for branch in BRANCHES:
                    branch_started = time.perf_counter()
                    sample = diagnostic.sample_arm(prepared, branch)
                    if branch in (BRANCHES[0], BRANCHES[3]):
                        check = diagnostic.verify_endpoint(prepared, branch, sample,
                            rtol=cfg['endpoint_rtol'], atol=cfg['endpoint_atol'])
                        result['endpoint_checks'].append(dict(case_id=case['case_id'], branch=branch, **check))
                    if case_index == 0 and branch == BRANCHES[3]:
                        result['raw_response_endpoint'] = dict(case_id=case['case_id'], branch=branch,
                            **raw_response_endpoint(model, prepared, count[None], stimulus[None], sample,
                                                    rtol=cfg['endpoint_rtol'], atol=cfg['endpoint_atol']))
                    expected_means, mean_consumed = conditional_count_means(sample.conditional_expected_counts, horizons)
                    paths = _sample_payload(sample, cohort.unit_ids, draws, h, bins,
                                            cfg['bin_ms'] / 1000, binding['config']['model'])
                    generated = sample.counts[:, 0]
                    del sample
                    projected, actual = _project_units(generated, observed,
                        cohort.unit_field_index, len(cohort.field_ids))
                    name = f"{case['case_id']}_{branch}.npz"
                    _saved_branch(diagnostics / name, dict(sampled_field_counts=projected,
                        observed_field_counts=actual, unit_ids=cohort.unit_ids, field_ids=cohort.field_ids,
                        unit_field_index=cohort.unit_field_index, sampling_seeds=np.asarray(seeds, dtype=np.int64),
                        known_input_sequence=stimulus, **paths))
                    first = case_index == 0
                    if first:
                        _saved_branch(diagnostics / ('full_unit_' + branch + '.npz'),
                                      dict(sampled_unit_counts=generated, unit_ids=cohort.unit_ids))
                    metrics, exact, consumed = numpy_unit_metrics(generated, observed, intervals,
                        data.bin_ticks, horizons, cfg['refractory_support_windows_bins'])
                    scaled = (projected / (metadata['unit_count'] * cfg['bin_ms'] / 1000) - mean) / sd
                    truth = (actual / (metadata['unit_count'] * cfg['bin_ms'] / 1000) - mean) / sd
                    scope = 'exact_saved_full_unit_array' if first else 'regenerated_counts_no_saved_full_unit_array'
                    branch_rows = []
                    for ms, length in zip(cfg['horizons_ms'], horizons):
                        row = indexed[(case['case_id'], branch, float(ms))]
                        branch_rows.extend(_metric_rows(metrics[length], exact[length], row,
                            family='unit', scope=scope, sampling_seeds=seeds))
                        independent_fields = replay_scores(scaled[:, :length], truth[:length], pairs,
                                                          cfg['temporal_lags_bins'])
                        for lag in cfg['temporal_lags_bins']:
                            if lag >= length:
                                independent_fields[f'variogram_temporal_lag_{lag}'] = None
                        branch_rows.extend(_metric_rows(independent_fields, {}, row,
                            family='field', scope=scope, sampling_seeds=seeds))
                        branch_rows.extend(_metric_rows(
                            dict(conditional_expected_spikes_per_unit=expected_means[length]), {}, row,
                            family='conditional_mean', scope=scope, sampling_seeds=seeds))
                    comparisons.extend(branch_rows)
                    with csv_path.open('a', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(branch_rows[0]))
                        if result['verified_branches'] == 0:
                            writer.writeheader()
                        writer.writerows(branch_rows)
                    if not all(row['passed'] for row in branch_rows):
                        result['status'] = 'MISMATCH'
                        raise ValueError('FROZEN_INDEPENDENT_METRIC_TOLERANCE_FAILED')
                    result['branch_replays'].append(dict(case_id=case['case_id'], branch=branch,
                        role=branch_role(branch)['role'], full_unit_reference=scope,
                        exact_field_and_lfads_paths=True, exact_identities_stimuli_and_sampling_seeds=True,
                        sampled_unit_count_values_consumed=consumed,
                        conditional_mean_values_consumed=mean_consumed,
                        elapsed_seconds=time.perf_counter() - branch_started))
                    result['verified_branches'] += 1
                    result['verified_score_rows'] += len(horizons)
                    _write_json(output, result)
                    print(json.dumps(dict(status='REPLAYED_LFADS_BOUNDARY_BRANCH', training_seed=binding['seed'],
                        case_id=case['case_id'], branch=branch, completed_branches=result['verified_branches'])), flush=True)
                    del generated, paths, projected, actual, scaled, truth, branch_rows
                result['verified_cases'] += 1
                _write_json(output, result)
                del prepared, count, stimulus, observed, data
        if (result['verified_cases'] != cfg['expected_cases_per_seed']
                or result['verified_branches'] != cfg['expected_cases_per_seed'] * len(BRANCHES)
                or result['verified_score_rows'] != len(indexed)
                or len(result['endpoint_checks']) != 2 * cfg['expected_cases_per_seed']
                or result['raw_response_endpoint'] is None):
            raise ValueError('COMPLETE_BOUNDARY_REPLAY_DENOMINATOR_REQUIRED')
        fields_checked = [r for r in comparisons if r['metric_family'] == 'field' and r['independent_value'] is not None]
        result.update(status='PASS', all_branch_unit_metrics_replayed=True, all_conditional_means_replayed=True,
            metric_comparisons=len(comparisons), field_scores_recomputed=len(fields_checked),
            maximum_field_absolute_difference=max(row['absolute_difference'] for row in fields_checked),
            saved_full_unit_branches_compared=len(BRANCHES),
            regenerated_branches_without_saved_full_unit_arrays=result['verified_branches'] - len(BRANCHES),
            sampled_unit_count_values_consumed=sum(row['sampled_unit_count_values_consumed'] for row in result['branch_replays']),
            conditional_mean_values_consumed=sum(row['conditional_mean_values_consumed'] for row in result['branch_replays']),
            csv_artifact=csv_path.name, csv_sha256=sha256_file(csv_path), elapsed_seconds=time.perf_counter() - started)
        _write_json(output, result)
        return result
    except BaseException as error:
        if result['status'] == 'RUNNING':
            result['status'] = 'FAILED'
        result.update(error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        if csv_path.exists():
            result.update(csv_artifact=csv_path.name, csv_sha256=sha256_file(csv_path))
        _write_json(output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition', 'fields', 'training', 'diagnostics', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--inference-migration', type=Path)
    args = parser.parse_args()
    preexisting = args.output.exists() or args.output.with_suffix('.csv').exists()
    try:
        result = replay(args.acquisition, args.fields, args.training, args.diagnostics, args.output,
                        args.device, inference_migration=args.inference_migration)
        print(json.dumps({key: result[key] for key in
            ('status', 'training_seed', 'verified_cases', 'verified_branches', 'verified_score_rows')}))
    except Exception as error:
        if not preexisting and args.output.suffix == '.json' and not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, dict(status='FAILED', error=str(error), error_type=type(error).__name__,
                                         traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

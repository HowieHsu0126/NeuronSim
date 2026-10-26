"""Replay the retained seed41 integer-bound failure without changing forecasts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .lfads_evaluator_compatibility import verify_legacy_evaluator
from .lfads_forecast_replay import (
    LEGACY_UNIT_METRIC_SHA256, MATCHED_PROTOCOL, METRIC_ATOL, METRIC_RTOL,
    LFADSRandomStreams, _exact_array, _loaded_model, _native_episode,
    _project_units, _same_execution_migration, _sample_payload, _score_rows,
    _verified_training, numpy_unit_metrics,
)
from .lfads_trajectory_training import _write_json
from .provenance import sha256_file

BOUND_ERROR = 'count residuals exceed exact int64 correction reduction support'
BOUNDED_METRIC_SHA256 = 'ffa2ed0f163549bb121f835432560fedf1eb734296e2adc36e800eb6f2cde8ad'
BOUNDED_CHUNK_UNITS = 4096
ORACLE_CHUNK_UNITS = 2048
INT64_LIMIT = (1 << 63) - 1


def next_failed_origin(failure, cases, origins):
    """The retained failure follows five complete origin-outer/case-inner steps."""
    if (failure.get('status') != 'FAILED' or failure.get('error') != BOUND_ERROR
            or failure.get('training_seed') != 41 or failure.get('verified_origins') != 5
            or not cases or not origins):
        raise ValueError('bound original seed41 failure after exactly five origins required')
    origin_index, case_index = divmod(failure['verified_origins'], len(cases))
    if origin_index >= len(origins):
        raise ValueError('failure is outside the declared replay order')
    return origin_index, case_index, cases[case_index], origins[origin_index]


def _pair_product_sum(left, right):
    """Independent direct dot product: bounded int64 blocks, Python extreme cells."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or left.shape != right.shape or not left.size
            or left.dtype != np.dtype('int64') or right.dtype != np.dtype('int64')
            or np.any(left == -(1 << 63)) or np.any(right == -(1 << 63))):
        raise ValueError('nonempty aligned signed residual vectors excluding INT64_MIN required')
    pending = [(left, right)]
    total = fixed_cells = python_cells = 0
    while pending:
        a, b = pending.pop()
        product_bound = int(np.abs(a).max()) * int(np.abs(b).max())
        if product_bound <= INT64_LIMIT:
            step = len(a) if product_bound == 0 else min(len(a), INT64_LIMIT // product_bound)
            for start in range(0, len(a), step):
                stop = min(start + step, len(a))
                total += int(np.sum(a[start:stop] * b[start:stop], dtype=np.int64))
                fixed_cells += stop - start
        elif len(a) > 1:
            middle = len(a) // 2
            pending.extend(((a[:middle], b[:middle]), (a[middle:], b[middle:])))
        else:
            total += int(a[0]) * int(b[0])
            python_cells += 1
    return total, fixed_cells, python_cells


def ordered_pair_oracle(samples, observed, *, chunk_units=ORACLE_CHUNK_UNITS):
    """Sum i<j residual products times two; do not reuse the candidate identity."""
    samples, observed = np.asarray(samples), np.asarray(observed)
    if (samples.ndim != 3 or min(samples.shape) < 1 or samples.shape[0] < 2
            or observed.shape != samples.shape[1:]
            or samples.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or observed.dtype.kind not in 'iuf'
            or np.any(samples < 0) or not np.isfinite(observed).all()
            or np.any(observed < 0) or np.any(observed != np.floor(observed))
            or int(observed.max()) > INT64_LIMIT
            or type(chunk_units) is not int or chunk_units < 1):
        raise ValueError('complete nonnegative signed samples and int64-representable integer observations required')
    draws, bins, units = samples.shape
    numerator = fixed_cells = python_cells = 0
    for start in range(0, units, chunk_units):
        residual = samples[:, :, start:start+chunk_units].astype(np.int64)
        residual -= observed[None, :, start:start+chunk_units].astype(np.int64)
        residual = residual.reshape(draws, -1)
        for i in range(draws):
            for j in range(i + 1, draws):
                value, fixed, arbitrary = _pair_product_sum(residual[i], residual[j])
                numerator += 2 * value
                fixed_cells += fixed
                python_cells += arbitrary
    expected_pairs = draws * (draws - 1) // 2 * bins * units
    if fixed_cells + python_cells != expected_pairs:
        raise ValueError('independent pair denominator incomplete')
    denominator = draws * (draws - 1) * bins * units
    value = numerator / denominator
    return dict(exact_numerator=numerator, exact_denominator=denominator,
                exact_rational_value=value, float32_value=float(np.float32(value)),
                unordered_pair_cell_products=expected_pairs,
                fixed_width_pair_cells=fixed_cells, python_integer_pair_cells=python_cells,
                method='direct i<j residual products times2; safe int64 chunks and extreme-cell Python products')


def conservative_bounds(samples, observed):
    """Reconstruct the frozen bounded implementation's actual rejection condition."""
    draws, bins, units = samples.shape
    blocks = []
    for start in range(0, units, BOUNDED_CHUNK_UNITS):
        stop = min(start + BOUNDED_CHUNK_UNITS, units)
        residual = samples[:, :, start:stop].astype(np.int64) - observed[None, :, start:stop].astype(np.int64)
        maximum = int(np.abs(residual).max())
        bound = draws * draws * maximum * maximum * bins * (stop - start)
        blocks.append(dict(unit_start=start, unit_stop=stop, maximum_absolute_residual=maximum,
                           conservative_bound=bound, exceeds_int64=bound > INT64_LIMIT))
    rejected = [row for row in blocks if row['exceeds_int64']]
    return dict(int64_limit=INT64_LIMIT, blocks=blocks,
                maximum_conservative_bound=max(row['conservative_bound'] for row in blocks),
                first_rejected_block=rejected[0] if rejected else None)


def _load_metrics(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('explicit metric source cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(value, reference):
    if reference is None:
        return dict(absolute_difference=None, allowed_difference=None, passed=value is None)
    if value is None or not np.isfinite(value) or not np.isfinite(reference):
        return dict(absolute_difference=None, allowed_difference=None, passed=False)
    delta = abs(float(value) - float(reference))
    allowed = METRIC_ATOL + METRIC_RTOL * abs(float(value))
    return dict(absolute_difference=delta, allowed_difference=allowed, passed=delta <= allowed)


def diagnose(acquisition, fields, training, forecasts, migration, candidate,
             bounded_candidate, failure, legacy_evaluator_source, output):
    acquisition, fields, training, forecasts, migration, candidate, bounded_candidate, failure, legacy_evaluator_source, output = map(
        Path, (acquisition, fields, training, forecasts, migration, candidate,
               bounded_candidate, failure, legacy_evaluator_source, output))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = None
    result = dict(status='RUNNING', training_seed=41, scientific_claim_qualified=False,
                  scope='one next origin from a bound retained range failure; no scientific fidelity qualification',
                  metric_tolerance=dict(rtol=METRIC_RTOL, atol=METRIC_ATOL),
                  count_selection='all draws, bins and identified units at every declared horizon',
                  comparisons=[], horizons=[], all_saved_fields_and_latent_paths_exact=False)
    _write_json(output/'report.json', result)
    try:
        inputs = dict(candidate=candidate, bounded_candidate=bounded_candidate, failure=failure,
                      legacy_evaluator_source=legacy_evaluator_source,
                      evaluation=forecasts/'evaluation.json', scores=forecasts/'scores.csv',
                      runtime_metric_source=Path(__file__).with_name('trajectory_sample_metrics.py'))
        hashes = {name: sha256_file(path) for name, path in inputs.items()}
        result.update(input_sha256=hashes, candidate_metric_sha256=hashes['candidate'],
                      bounded_metric_sha256=hashes['bounded_candidate'],
                      implementation_sha256=sha256_file(Path(__file__)))
        if hashes['bounded_candidate'] != BOUNDED_METRIC_SHA256:
            raise ValueError('pinned rejected bounded metric source required')
        ev = json.loads(inputs['evaluation'].read_text())
        failed = json.loads(failure.read_text())
        if (ev['status'] != 'EVALUATED_DEVELOPMENT' or ev['model_type'] != 'lfads'
                or ev['training_seed'] != 41 or ev['configuration'] != MATCHED_PROTOCOL
                or ev['unit_metrics_implementation_sha256'] != LEGACY_UNIT_METRIC_SHA256
                or hashes['runtime_metric_source'] != LEGACY_UNIT_METRIC_SHA256
                or ev['scores_sha256'] != hashes['scores']
                or failed.get('evaluation_sha256') != hashes['evaluation']
                or failed.get('scores_sha256') != hashes['scores']
                or failed.get('dataset_sha256') != ev['dataset_sha256']
                or failed.get('model_type') != 'lfads'
                or failed.get('metric_tolerance') != result['metric_tolerance']):
            raise ValueError('original evaluation, score table, legacy runtime and failure bindings required')
        cohort, run, directory, selected, loaded_migration = _verified_training(acquisition, fields, training, migration)
        binding = run['binding']
        execution = {}
        if (binding != ev['training_binding'] or binding['seed'] != 41
                or ev['training_run_sha256'] != sha256_file(training/'run.json')
                or ev['selected_training_sha256'] != sha256_file(directory/'training.json')
                or selected['model_artifacts'] != ev['selected_model_artifacts']
                or not _same_execution_migration(ev['inference_migration'], loaded_migration, run, execution)):
            raise ValueError('unchanged failed-case training, checkpoint and migration bridge required')
        current_evaluator = Path(__file__).with_name('lfads_trajectory_evaluation.py')
        if ev['evaluator_implementation_sha256'] == sha256_file(current_evaluator):
            execution['execution_evaluator_sha256'] = ev['evaluator_implementation_sha256']
        else:
            execution['verified_legacy_evaluator'] = verify_legacy_evaluator(
                ev['evaluator_implementation_sha256'], legacy_evaluator_source, current_evaluator)
        cfg = ev['configuration']
        row_record = dict(ev, artifacts={k:v for k,v in ev['artifacts'].items() if k != 'fitting_field_scale.npz'})
        _score_rows(forecasts, row_record, cohort.validation)
        origin_index, case_index, case, origin = next_failed_origin(failed, cohort.validation, cfg['origins_ms'])
        result.update(original_failure=failed, execution_compatibility=execution,
                      original_unit_metrics_sha256=ev['unit_metrics_implementation_sha256'],
                      case_id=case['case_id'], case_index=case_index, origin_index=origin_index,
                      origin_ms=origin, original_replay_sha256=hashes['failure'])
        bin_ms = binding['config']['bin_ms']
        horizons = [int(ms/bin_ms) for ms in cfg['horizons_ms']]
        boundary, longest, draws = int(origin/bin_ms), max(horizons), cfg['joint_samples']
        if boundary*bin_ms != origin or any(h*bin_ms != ms for h,ms in zip(horizons,cfg['horizons_ms'])):
            raise ValueError('exact source-clock alignment required')
        data = _native_episode(cohort, case)
        if data.source_identity != ev['native_source_identity'] or data.bin_ms != bin_ms:
            raise ValueError('native source or observation clock changed')
        counts, stimulus = data.window(0, data.duration_ticks)
        if boundary < 1 or boundary + longest > len(counts):
            raise ValueError('complete causal prefix and forecast support required')
        model = _loaded_model(directory, cohort, binding, boundary, longest, 'cuda')
        state = model.infer_state(dict(counts=counts[None,:boundary], stimulus=stimulus[None,:boundary]),
            metadata=dict(unit_ids=list(map(int,cohort.unit_ids)), bin_seconds=bin_ms/1000))
        seed = cfg['sampling_seed_base'] + 41*100000 + case_index*100 + origin_index*3
        result['sampling_seeds'] = dict(initial=seed, dynamics=seed+1, observation=seed+2)
        sample = model.simulate(state, stimulus[None,boundary:boundary+longest],
            LFADSRandomStreams.from_seeds(seed,seed+1,seed+2), draws)
        payload = _sample_payload(sample,cohort.unit_ids,draws,boundary,longest,bin_ms/1000,binding['config']['model'])
        generated, observed = sample.counts[:,0], counts[boundary:boundary+longest]
        del sample
        if generated.shape != (draws,longest,len(cohort.unit_ids)) or observed.shape != generated.shape[1:]:
            raise ValueError('full identified count denominator changed')
        projected, truth = _project_units(generated, observed, cohort.unit_field_index, len(cohort.field_ids))
        name = f"{case['case_id']}_origin_{origin:g}.npz"
        if Path(name).name != name or sha256_file(forecasts/name) != ev['artifacts'][name]:
            raise ValueError('saved selected-origin artifact changed')
        with np.load(forecasts/name,allow_pickle=False) as saved:
            if set(saved.files) != {'sampled_field_counts','observed_field_counts',*payload}:
                raise ValueError('saved field/path schema changed')
            for key,value in [('sampled_field_counts',projected),('observed_field_counts',truth),*payload.items()]:
                _exact_array(saved[key],value,'unchanged saved fields and latent payload required: '+key)
        result.update(all_saved_fields_and_latent_paths_exact=True, saved_origin_artifact_sha256=ev['artifacts'][name],
                      full_unit_sample_scope='new diagnostic retention; this origin has no preexisting saved full-unit array')
        bounded = _load_metrics(bounded_candidate,'bounded_count_metrics_for_range_diagnostic')
        torch.set_num_threads(binding['config']['cpu_threads'])
        rejected_horizons = []
        # Establish every rejected horizon and retain the real inputs before
        # loading/executing the adaptive candidate or either independent oracle.
        for ms,h in zip(cfg['horizons_ms'],horizons):
            target = torch.from_numpy(observed[:h]).float()
            if not np.array_equal(target.double().numpy(), observed[:h].astype(np.float64)):
                raise ValueError('legacy float32 observation cast loses actual integer counts')
            args = (torch.from_numpy(generated[:,:h]),target,data.minimum_interval_ticks,
                    data.bin_ticks,cfg['refractory_support_windows_bins'])
            bounds = conservative_bounds(generated[:,:h],observed[:h])
            row = dict(horizon_ms=ms,bins=h,sampled_count_max=int(generated[:,:h].max()),
                       observed_count_max=int(observed[:h].max()),bounds=bounds,bounded_status='RUNNING')
            result['horizons'].append(row)
            old = None
            try:
                old = bounded.unit_count_metrics(*args)
                row.update(bounded_status='RETURNED', bounded_metrics=old)
            except ValueError as error:
                if str(error) != BOUND_ERROR:
                    raise
                row.update(bounded_status='REJECTED_CONSERVATIVE_BOUND', bounded_error=str(error))
                rejected_horizons.append(h)
            if (old is None) != (bounds['first_rejected_block'] is not None):
                raise ValueError('pinned bounded implementation and independent bound disagree')
        if not rejected_horizons:
            raise ValueError('original conservative range failure did not reproduce')
        longest_failed = max(rejected_horizons)
        np.savez_compressed(output/'failed_case_counts.npz',sampled_counts=generated[:,:longest_failed],
            observed_counts=observed[:longest_failed],unit_ids=cohort.unit_ids,
            minimum_interval_ticks=data.minimum_interval_ticks,origin_ms=origin,
            horizon_ms=longest_failed*bin_ms,initial_seed=seed,dynamics_seed=seed+1,observation_seed=seed+2)
        result.update(counts_sha256=sha256_file(output/'failed_case_counts.npz'),
                      retained_horizon_ms=longest_failed*bin_ms,rejected_horizons_ms=[h*bin_ms for h in rejected_horizons],
                      sampled_integer_count_values=generated[:,:longest_failed].size,
                      observed_integer_count_values=observed[:longest_failed].size)
        _write_json(output/'report.json',result)
        independent, _, consumed = numpy_unit_metrics(generated,observed,data.minimum_interval_ticks,
            data.bin_ticks,horizons,cfg['refractory_support_windows_bins'])
        result['regenerated_count_values'] = consumed
        revised = _load_metrics(candidate,'adaptive_count_metrics_for_range_diagnostic')
        for row in result['horizons']:
            ms,h = row['horizon_ms'],row['bins']
            args = (torch.from_numpy(generated[:,:h]),torch.from_numpy(observed[:h]).float(),
                    data.minimum_interval_ticks,data.bin_ticks,cfg['refractory_support_windows_bins'])
            old = row.get('bounded_metrics')
            new = revised.unit_count_metrics(*args)
            row['candidate_metrics'] = new
            if set(new) != set(independent[h]) or (old is not None and set(old) != set(new)):
                raise ValueError('unit metric schema changed')
            for metric,value in new.items():
                comparison = dict(horizon_ms=ms,metric=metric,revised=value,
                                  independent_float64=independent[h][metric],**_comparison(value,independent[h][metric]))
                if old is not None:
                    comparison['bounded_value'] = old[metric]
                    comparison['unchanged_bounded_value'] = value == old[metric]
                result['comparisons'].append(comparison)
            if old is None:
                oracle = ordered_pair_oracle(generated[:,:h],observed[:h])
                row['ordered_pair_oracle'] = oracle
                if new['unit_bin_mean_mse_corrected'] != oracle['float32_value']:
                    raise ValueError('adaptive corrected MSE differs from independent exact pair oracle')
                oracle_comparison = _comparison(oracle['float32_value'],independent[h]['unit_bin_mean_mse_corrected'])
                row['oracle_numpy_comparison'] = oracle_comparison
                if not oracle_comparison['passed']:
                    raise ValueError('exact pair oracle and NumPy reference violate original tolerance')
            _write_json(output/'report.json',result)
        if (not all(r['passed'] for r in result['comparisons'])
                or not all(r.get('unchanged_bounded_value',True) for r in result['comparisons'])):
            raise ValueError('candidate changes a returned bounded score or violates original NumPy tolerance')
        for key,path in inputs.items():
            if sha256_file(path) != hashes[key]:
                raise ValueError('diagnostic input changed during execution: '+key)
        result['status'] = 'REPRODUCED_CONSERVATIVE_INT64_BOUND_EXACT_CORRECTION_PASS'
    except BaseException as error:
        result.update(status='FAILED',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        raise
    finally:
        try:
            if model is not None:
                model.close()
        except BaseException as error:
            result.update(status='FAILED',model_close_error=str(error))
            raise
        finally:
            result['elapsed_seconds'] = time.perf_counter()-started
            _write_json(output/'report.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition','fields','training','forecasts','migration','candidate',
                 'bounded-candidate','failure','legacy-evaluator-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = diagnose(args.acquisition,args.fields,args.training,args.forecasts,args.migration,
        args.candidate,args.bounded_candidate,args.failure,args.legacy_evaluator_source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','horizons','original_failure')},indent=2))


if __name__ == '__main__':
    main()

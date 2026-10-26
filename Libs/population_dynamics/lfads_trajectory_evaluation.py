"""Matched full-unit LFADS forecasts on the shared development selection blocks.

Each origin has one graph lifetime and uses its complete observed prefix. The
public LFADS sampler returns conditional means as well as identified counts;
that work is included in generation cost. Selection blocks support descriptive
comparisons, not independent efficacy, equivalence or convergence claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
import yaml

from .lfads import LFADSConditionalSimulator, LFADSRandomStreams, SOURCE_REVISION
from .lfads_trajectory_training import (
    TrajectoryCohort, CONTINUATION_SCHEMA, LEGACY_TRAINER_SHA256, _checkpoint_budget, _binding, _verify_checkpoint, _verified_run, _write_json, validate_config,
)
from .native_renewal_data import NativeRenewalEpisode
from .poisson_trajectory_evaluation import fitting_scale
from .provenance import sha256_file
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score, field_pairs, variogram_score


LEGACY_ADAPTER_SHA256 = '083ae71a344a1cf8c931c2ae1479d88c36c9bb5fddf64b6b05218f6451f5d9f7'
BUDGET_TRAINER_SHA256 = '8f29d35df1014be4bc98c1be43972524ce823c8b54dc1483d0c2e07e45bcb53f'
MATCHED_PROTOCOL = dict(
    scope='development_on_shared_epoch_selection_blocks',
    origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.],
    joint_samples=16, sampling_seed_base=86000, temporal_lags_bins=[1, 4, 16],
    refractory_support_windows_bins=[1, 2, 4],
    causal_history='all_native_bins_before_origin',
    field_scale='fitting_block_pooled_standard_deviation_of_per_unit_hz',
    finite_ensemble_corrections=True,
    matched_output='all_identified_unit_counts_and_identical_field_projection',
    unit_metric_dtype='float32', field_metric_dtype='float64',
    retain_full_unit_samples='first_validation_case_and_first_origin_per_seed_and_model',
    likelihood_comparison='omitted_observation_spaces_differ', formal_evaluation=False,
)


def _verify_inference_binding(saved, current, migration_path, *, training_continuation=None):
    """Verify numerical migration, with the pinned budget-trainer metadata bridge."""
    if saved == current:
        if migration_path is not None:
            raise ValueError('inference migration supplied but no source migration is required')
        return None
    old, new = saved.get('sources', {}), current.get('sources', {})
    trainer = 'lfads_trajectory_training.py'
    saved_trainer, current_trainer = old.get(trainer), new.get(trainer)
    legacy_to_budget = (saved_trainer == LEGACY_TRAINER_SHA256
                        and current_trainer == BUDGET_TRAINER_SHA256)
    expected_sources = dict(new, **{'lfads.py': old.get('lfads.py')})
    if legacy_to_budget:
        expected_sources[trainer] = LEGACY_TRAINER_SHA256
    expected = dict(current, sources=expected_sources)
    if (saved != expected or old.get('lfads.py') != LEGACY_ADAPTER_SHA256
            or new.get('lfads.py') == old.get('lfads.py') or migration_path is None):
        raise ValueError('LFADS evaluation config/data/source/backend binding changed')
    migration_path = Path(migration_path)
    receipt = json.loads(migration_path.read_text())
    unchanged = {key: value for key, value in old.items() if key != 'lfads.py'}
    receipt_sources = receipt.get('unchanged_sources')
    bridge = None
    if legacy_to_budget:
        bridge = dict(schema='lfads_pinned_budget_trainer_metadata_bridge_v1',
            legacy_trainer_sha256=LEGACY_TRAINER_SHA256, budget_trainer_sha256=BUDGET_TRAINER_SHA256,
            saved_trainer_sha256=saved_trainer, inference_trainer_sha256=current_trainer,
            receipt_trainer_sha256=LEGACY_TRAINER_SHA256, checkpoint_scope='legacy_parent',
            scope='trainer metadata only; numerical adapter migration evidence remains unchanged')
    elif receipt_sources != unchanged:
        # The original adapter migration was qualified beside the old trainer.
        # An explicit child lineage carries the reviewed budget trainer; its
        # frozen file and actual executed budget are checked by _verified_run.
        parent_binding = training_continuation.get('parent_binding', {}) if isinstance(training_continuation, dict) else {}
        parent_sources = parent_binding.get('sources', {})
        expected_parent = dict(saved, sources=dict(old, **{trainer: parent_sources.get(trainer)}))
        expected_change = ({trainer: dict(parent=LEGACY_TRAINER_SHA256, current=BUDGET_TRAINER_SHA256)}
                           if parent_sources.get(trainer) == LEGACY_TRAINER_SHA256 else {})
        if (saved_trainer != BUDGET_TRAINER_SHA256 or current_trainer != BUDGET_TRAINER_SHA256
                or not isinstance(training_continuation, dict)
                or training_continuation.get('schema') != CONTINUATION_SCHEMA
                or training_continuation.get('treatment') != 'optimization_budget_only'
                or training_continuation.get('current_binding') != saved
                or training_continuation.get('current_source_hashes') != old
                or parent_sources.get(trainer) not in (LEGACY_TRAINER_SHA256, BUDGET_TRAINER_SHA256)
                or parent_binding != expected_parent
                or training_continuation.get('parent_source_hashes') != parent_sources
                or training_continuation.get('source_migration') != expected_change):
            raise ValueError('generation-memory inference migration lacks the pinned continued-trainer lineage')
        unchanged = dict(unchanged, **{trainer: LEGACY_TRAINER_SHA256})
        bridge = dict(schema='lfads_pinned_budget_trainer_metadata_bridge_v1',
            legacy_trainer_sha256=LEGACY_TRAINER_SHA256, budget_trainer_sha256=BUDGET_TRAINER_SHA256,
            saved_trainer_sha256=saved_trainer, inference_trainer_sha256=current_trainer,
            receipt_trainer_sha256=LEGACY_TRAINER_SHA256, checkpoint_scope='continued_child',
            scope='trainer metadata only; frozen continuation and numerical adapter evidence remain required')
    if (receipt.get('schema') != 'lfads_generation_memory_inference_migration_v1'
            or receipt.get('status') != 'QUALIFIED_INFERENCE_ONLY'
            or receipt.get('from_adapter_sha256') != old['lfads.py']
            or receipt.get('to_adapter_sha256') != new['lfads.py']
            or receipt.get('source_revision') != SOURCE_REVISION
            or receipt_sources != unchanged):
        raise ValueError('generation-memory inference migration is not qualified for these exact sources')
    evidence = receipt.get('evidence', [])
    if {row.get('kind') for row in evidence} != {'cpu_equivalence', 'full_population_cuda'}:
        raise ValueError('CPU equivalence and full-population CUDA migration evidence required')
    for row in evidence:
        path = Path(row['path'])
        if not path.is_absolute():
            path = migration_path.parent / path
        if sha256_file(path) != row['sha256']:
            raise ValueError('qualified inference migration evidence changed')
    result = dict(receipt=receipt, receipt_sha256=sha256_file(migration_path))
    if bridge is not None:
        result['trainer_bridge'] = bridge
    return result



def _verified_training(acquisition, fields, training, migration_path=None):
    run = json.loads((training / 'run.json').read_text())
    if run.get('status') != 'FITTED_DEVELOPMENT':
        raise ValueError('terminal FITTED_DEVELOPMENT LFADS training required')
    saved = run['binding']
    config, seed = saved['config'], saved['seed']
    validate_config(config, seed)
    cohort = TrajectoryCohort(acquisition, fields, config)
    migration = _verify_inference_binding(saved, _binding(cohort, config, seed), migration_path,
        training_continuation=run.get('continuation'))
    checked_run, latest_directory, latest, selected_directory, selected = _verified_run(training, len(cohort.fit))
    if checked_run != run:
        raise ValueError('LFADS training run changed during evaluation validation')
    if migration is not None and 'trainer_bridge' in migration:
        migration['trainer_bridge']['verified_continuation_sha256'] = run.get('continuation_sha256')
    model = json.loads((selected_directory / 'model.json').read_text())
    expected = dict(unit_ids=list(map(int, cohort.unit_ids)), input_size=1,
        history_bins=config['history_bins'], horizon_bins=config['horizon_bins'],
        training_history_bins=config['history_bins'], training_horizon_bins=config['horizon_bins'],
        model_config=config['model'], initialization_seed=seed, bin_seconds=config['bin_ms'] / 1000,
        source_revision=SOURCE_REVISION, training_updates=selected['training_updates'])
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError('selected LFADS model configuration, unit identities or training boundary changed')
    return cohort, run, selected_directory, selected, migration


def _native_episode(cohort, case, source=None, intervals=None):
    cohort._verify_metadata(case)
    data = NativeRenewalEpisode(cohort.acquisition / case['case_id'], cohort.fields / case['case_id'],
        expected_source=source, expected_unit_ids=cohort.unit_ids,
        expected_unit_field_index=cohort.unit_field_index)
    cohort._verify_episode(data)
    if (any(data.source_identity.get(key) != value for key, value in cohort.source_identity.items())
            or not np.array_equal(data.field_ids, cohort.field_ids)
            or (intervals is not None and not np.array_equal(data.minimum_interval_ticks, intervals))):
        raise ValueError('evaluation native source, field identities or refractory law changed')
    row = cohort.records[case['case_id']]
    if (data.binding['episode_report_sha256'] != row['raw_report_sha256']
            or data.binding['field_summary_sha256'] != row['field_summary_sha256']):
        raise ValueError('evaluation episode differs from its canonical field-table row')
    return data


def _loaded_model(directory, cohort, binding, boundary, longest, device, *, prior_parameters=None):
    model = LFADSConditionalSimulator.load(directory, history_bins=boundary,
        horizon_bins=longest, device=device)
    config = binding['config']
    try:
        if (not model.loaded_for_inference or tuple(model.unit_ids) != tuple(map(int, cohort.unit_ids))
                or model.history_bins != boundary or model.horizon_bins != longest
                or model.training_history_bins != config['history_bins']
                or model.training_horizon_bins != config['horizon_bins']
                or model.model_config != config['model'] or model.initialization_seed != binding['seed']
                or model.input_size != 1 or model.bin_seconds != config['bin_ms'] / 1000):
            raise ValueError('LFADS inference view changed the trained model or unit identities')
        cohort.verify_model(model)
        if prior_parameters is not None:
            # The refitter reuses this module's training verifier; defer this
            # import to avoid a circular module initialization.
            from .lfads_prior_refit import apply_prior_parameters
            model.prior_refit_state_check = apply_prior_parameters(model, prior_parameters)
        return model
    except BaseException:
        model.close()
        raise


def _sample_payload(sample, units, draws, boundary, longest, bin_seconds, model_config):
    expected = (draws, 1, longest, len(units))
    if (tuple(sample.unit_ids) != tuple(map(int, units)) or sample.bin_seconds != bin_seconds
            or sample.counts.shape != expected or sample.counts.dtype != np.int64
            or np.any(sample.counts < 0) or sample.conditional_expected_counts.shape != expected
            or not np.isfinite(sample.conditional_expected_counts).all()
            or np.any(sample.conditional_expected_counts < 0)):
        raise ValueError('LFADS full-unit sample identity, shape, clock or count support changed')
    shapes = dict(initial_latent_states=(draws, 1, model_config['ic_dim']),
        historical_controller_inputs=(draws, 1, boundary, model_config['co_dim']),
        future_controller_inputs=(draws, 1, longest, model_config['co_dim']),
        boundary_generator_states=(draws, 1, model_config['gen_dim']),
        future_generator_states=(draws, 1, longest, model_config['gen_dim']))
    payload = {}
    for key, shape in shapes.items():
        value = getattr(sample, key)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('LFADS controller/generator sample shape or finite values changed: ' + key)
        payload['lfads_' + key] = value[:, 0]
    return payload


@torch.no_grad()
def _project_counts(generated, observed, mapping, field_count):
    # CPU integer projection is identical to the existing shared evaluator;
    # Torch never allocates on the TensorFlow device.
    index = torch.from_numpy(mapping)
    counts = torch.from_numpy(generated)
    truth = torch.from_numpy(observed)
    projected = torch.zeros((*counts.shape[:-1], field_count), dtype=torch.int64)
    actual = torch.zeros((*truth.shape[:-1], field_count), dtype=torch.int64)
    projected.index_add_(-1, index, counts)
    actual.index_add_(-1, index, truth.to(torch.int64))
    return projected.numpy(), actual.numpy()


def evaluate(acquisition, fields, training, config_path, output, device_name='cuda', *, inference_migration=None,
             prior_refit=None, conditional_prior=None, continue_from=None):
    started = time.perf_counter()
    acquisition, fields, training, config_path, output = map(Path, (acquisition, fields, training, config_path, output))
    if output.exists():
        raise FileExistsError('LFADS evaluation output already exists')
    config = yaml.safe_load(config_path.read_text())
    if config != MATCHED_PROTOCOL:
        raise ValueError('exact common development forecast protocol required')
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('explicit cpu or cuda LFADS execution required')
    if device_name == 'cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise ValueError('qualified LFADS CUDA execution requires NVIDIA_TF32_OVERRIDE=0')
    cohort, run, directory, selected, migration = _verified_training(acquisition, fields, training, inference_migration)
    prior_parameters = None
    if prior_refit is not None:
        from .lfads_prior_refit import load_prior_parameters
        prior_parameters, prior_binding = load_prior_parameters(prior_refit, run=run, selected=selected,
            cohort_binding=cohort.binding, training_run_sha256=sha256_file(training / 'run.json'))
    mean_prior = None
    if conditional_prior is not None:
        if prior_refit is None:
            raise ValueError('conditional mean contrast requires the verified fitted AR prior')
        from .lfads_conditional_prior import load_mean_prior
        mean_prior, mean_binding = load_mean_prior(conditional_prior, prior_binding,
            training_seed=run['binding']['seed'],
            network_budget_epochs=_checkpoint_budget(run, run['binding']['config'])[0],
            selected_model_artifacts=selected['model_artifacts'])
    binding, fit_config = run['binding'], run['binding']['config']
    native = _native_episode(cohort, cohort.validation[0])
    intervals, source = native.minimum_interval_ticks.copy(), native.source_identity
    bin_ms = fit_config['bin_ms']
    horizons = [int(ms / bin_ms) for ms in config['horizons_ms']]
    boundaries = [int(ms / bin_ms) for ms in config['origins_ms']]
    if (any(h * bin_ms != ms or h < 1 for h, ms in zip(horizons, config['horizons_ms']))
            or any(h * bin_ms != ms or h < 1 for h, ms in zip(boundaries, config['origins_ms']))
            or max(boundaries) + max(horizons) > native.duration_ticks // native.bin_ticks):
        raise ValueError('complete aligned causal prefixes and forecast support required')
    scale_mean, scale_sd, metadata = fitting_scale(fields, cohort.fit)
    if (not np.array_equal(native.field_ids, metadata['field_ids'])
            or not np.array_equal(np.bincount(native.unit_field_index, minlength=len(native.field_ids)), metadata['unit_count'])):
        raise ValueError('evaluation field identities or denominators differ from fitting')
    definitions = [json.loads(str(value)) for value in metadata['field_ids']]
    pairs = field_pairs([r[0] for r in definitions], [r[2] for r in definitions], [r[1] for r in definitions])
    torch.set_num_threads(fit_config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    longest, draws = max(horizons), config['joint_samples']
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', model_type='lfads', configuration=config, training_seed=binding['seed'],
        dataset_sha256=binding['dataset_sha256'], acquisition_design_sha256=binding['acquisition_design_sha256'],
        training_binding=binding, training_run_sha256=sha256_file(training / 'run.json'),
        selected_checkpoint=run['best_checkpoint'], selected_training_sha256=sha256_file(directory / 'training.json'),
        selected_model_artifacts=selected['model_artifacts'], selected_epoch=run['best_epoch'],
        completed_training_epochs=run['completed_epochs'], training_stop_reason=run['stop_reason'],
        effective_training_epoch_budget=run.get('effective_maximum_epochs', fit_config['maximum_epochs']),
        training_continuation=run.get('continuation'),
        evaluation_blocks=fit_config['validation_blocks'], inference_migration=migration,
        all_unit_outputs_generated=True, field_outputs_from_unit_counts=True,
        scientific_claim_qualified=False, formal_test_accessed=False, calibration_outcomes_accessed=False,
        convergence_established=False, native_source_identity=source, likelihood_scores_compared=False,
        output_contract='identified unit counts; public sampler also computes conditional means',
        sampling_cost_scope='public simulate including conditional means, controller ancestry and all unit counts',
        unit_metric_estimator='float32 CPU Torch estimator shared with Poisson/renewal matched evaluation',
        field_metric_estimator='float64 fitting-normalized finite-ensemble scores',
        peak_tensorflow_gpu_allocated_bytes=None, gpu_memory_measurement='not measured; Torch peaks are not TensorFlow peaks',
        host_memory_measurement='Linux process-lifetime ru_maxrss; includes graph, training weights, arrays and scoring',
        full_unit_replay_scope='first validation case and first origin; other origins retain exact fields and LFADS sample paths',
        graph_lifetimes=[], completed_origins=0, score_rows=0,
        evaluator_implementation_sha256=sha256_file(Path(__file__)),
        unit_metrics_implementation_sha256=sha256_file(Path(__file__).with_name('trajectory_sample_metrics.py')),
        verification_and_scale_seconds=time.perf_counter() - started)
    if prior_refit is not None:
        record['prior_refit'] = prior_binding
    if mean_prior is not None:
        record['model_variant'] = 'template_time_controller_mean_prior'
        record['conditional_prior'] = mean_binding
    _write_json(output / 'evaluation.json', record)
    artifacts, row_count = {}, 0
    np.savez(output / 'fitting_field_scale.npz', mean_hz=scale_mean, standard_deviation_hz=scale_sd, **metadata)
    artifacts['fitting_field_scale.npz'] = sha256_file(output / 'fitting_field_scale.npz')
    del native
    completed_keys = None
    if continue_from is not None:
        from .lfads_forecast_continuation import prepare_continuation
        completed_keys, artifacts, row_count = prepare_continuation(
            continue_from, output, record, cohort.validation, Path(__file__))
        _write_json(output / 'evaluation.json', record)
    try:
        for origin_index, (origin, boundary) in enumerate(zip(config['origins_ms'], boundaries)):
            if completed_keys is not None:
                if all((case['case_id'], origin) in completed_keys for case in cohort.validation):
                    continue
            model = None
            graph_started = time.perf_counter()
            try:
                model = _loaded_model(directory, cohort, binding, boundary, longest, device_name,
                    prior_parameters=prior_parameters)
                graph_seconds = time.perf_counter() - graph_started
                graph = dict(origin_ms=origin, history_bins=boundary, horizon_bins=longest,
                    setup_seconds=graph_seconds, execution_device=model.execution_device, completed_cases=0)
                if prior_refit is not None:
                    graph['prior_refit_state_check'] = model.prior_refit_state_check
                record['graph_lifetimes'].append(graph)
                _write_json(output / 'evaluation.json', record)
                for case_index, case in enumerate(cohort.validation):
                    if completed_keys is not None:
                        if (case['case_id'], origin) in completed_keys:
                            continue
                    origin_started = time.perf_counter()
                    data = _native_episode(cohort, case, source, intervals)
                    count, stimulus = data.window(0, data.duration_ticks)
                    if (count.shape != (1200, len(cohort.unit_ids)) or stimulus.shape != (1200, 1)
                            or count.dtype not in (np.float32, np.int32, np.int64)):
                        raise ValueError('complete native episode counts and actual TC control sequence required')
                    load_seconds = time.perf_counter() - origin_started
                    t = time.perf_counter()
                    # The state receives no response at or after the origin.
                    state = model.infer_state(dict(counts=count[None, :boundary], stimulus=stimulus[None, :boundary]),
                        metadata=dict(unit_ids=list(map(int, cohort.unit_ids)), bin_seconds=bin_ms / 1000))
                    inference_seconds = time.perf_counter() - t
                    seed_base = config['sampling_seed_base'] + binding['seed'] * 100000 + case_index * 100 + origin_index * 3
                    streams = LFADSRandomStreams.from_seeds(seed_base, seed_base + 1, seed_base + 2)
                    t = time.perf_counter()
                    if mean_prior is None:
                        sample = model.simulate(state, stimulus[None, boundary:boundary + longest], streams, draws)
                    else:
                        from .lfads_conditional_prior import simulate_with_mean
                        sample = simulate_with_mean(model, state, stimulus[None, boundary:boundary + longest], streams, draws,
                            mean_prior.curve(case['template_id'], boundary, longest))
                    generation_seconds = time.perf_counter() - t
                    paths = _sample_payload(sample, cohort.unit_ids, draws, boundary, longest, bin_ms / 1000, fit_config['model'])
                    if mean_prior is not None:
                        paths['lfads_base_future_controller_inputs'] = sample.base_future_controller_inputs[:, 0]
                    generated, observed = sample.counts[:, 0], count[boundary:boundary + longest]
                    del sample  # Conditional means are computed but are not requested score outputs.
                    t = time.perf_counter()
                    raw_fields, raw_truth = _project_counts(generated, observed, cohort.unit_field_index, len(cohort.field_ids))
                    projection_seconds = time.perf_counter() - t
                    sampled = (raw_fields / (metadata['unit_count'] * bin_ms / 1000) - scale_mean) / scale_sd
                    truth = (raw_truth / (metadata['unit_count'] * bin_ms / 1000) - scale_mean) / scale_sd
                    name = f"{case['case_id']}_origin_{origin:g}.npz"
                    t = time.perf_counter()
                    np.savez_compressed(output / name, observed_field_counts=raw_truth,
                        sampled_field_counts=raw_fields, **paths)
                    artifacts[name] = sha256_file(output / name)
                    if case_index == origin_index == 0:
                        np.savez_compressed(output / 'full_unit_samples.npz', sampled_unit_counts=generated,
                            unit_ids=cohort.unit_ids, initial_ready_in_ticks=data.ready_at(boundary * data.bin_ticks), **paths)
                        artifacts['full_unit_samples.npz'] = sha256_file(output / 'full_unit_samples.npz')
                    serialization_seconds = time.perf_counter() - t
                    t = time.perf_counter()
                    rows = []
                    # The established estimator consumes float32 observations;
                    # integer samples stay on the CPU and retain exact projection.
                    generated_tensor = torch.from_numpy(generated)
                    observed_tensor = torch.from_numpy(observed).to(dtype=torch.float32)
                    for ms, horizon in zip(config['horizons_ms'], horizons):
                        row = dict(seed=binding['seed'], block=case['block'], case_id=case['case_id'],
                            template_id=case['template_id'], origin_ms=origin, horizon_ms=ms, method='lfads',
                            energy_score=energy_score(sampled[:, :horizon], truth[:horizon]),
                            **unit_count_metrics(generated_tensor[:, :horizon], observed_tensor[:horizon], intervals,
                                data.bin_ticks, config['refractory_support_windows_bins']),
                            inference_seconds=inference_seconds, full_unit_generation_seconds=generation_seconds,
                            field_projection_seconds=projection_seconds, input_loading_seconds=load_seconds,
                            sample_serialization_seconds=serialization_seconds, graph_setup_seconds=graph_seconds,
                            peak_tensorflow_gpu_allocated_bytes=None, ensemble_artifact=name)
                        for label, pair in pairs.items():
                            row['variogram_' + label] = variogram_score(sampled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
                        for lag in config['temporal_lags_bins']:
                            row[f'variogram_temporal_lag_{lag}'] = (variogram_score(sampled[:, :horizon], truth[:horizon],
                                np.arange(len(cohort.field_ids)), np.arange(len(cohort.field_ids)), lag) if lag < horizon else None)
                        rows.append(row)
                    scoring_seconds = time.perf_counter() - t
                    for row in rows:
                        row.update(scoring_seconds=scoring_seconds, origin_total_seconds=time.perf_counter() - origin_started,
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                    with (output / 'scores.csv').open('a', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        if row_count == 0:
                            writer.writeheader()
                        writer.writerows(rows)
                    row_count += len(rows)
                    graph['completed_cases'] += 1
                    record.update(completed_origins=record['completed_origins'] + 1, score_rows=row_count)
                    _write_json(output / 'evaluation.json', record)
                    print(json.dumps(dict(status='EVALUATING_MATCHED_COUNTS', model='lfads', case=case['case_id'],
                        origin_ms=origin, rows=row_count)), flush=True)
                    del generated, observed, generated_tensor, observed_tensor, paths, state, count, stimulus, data
            finally:
                if model is not None:
                    model.close()
                    model = None
        expected_origins = len(cohort.validation) * len(boundaries)
        if record['completed_origins'] != expected_origins or row_count != expected_origins * len(horizons):
            raise ValueError('LFADS origin/sample/score protocol is incomplete')
        record.update(status='EVALUATED_DEVELOPMENT', score_rows=row_count, episodes=len(cohort.validation),
            independent_blocks=len({case['block'] for case in cohort.validation}), artifacts=artifacts,
            scores_sha256=sha256_file(output / 'scores.csv'), elapsed_seconds=time.perf_counter() - started,
            inference='descriptive epoch-selection blocks; not independent efficacy or equivalence evidence')
        _write_json(output / 'evaluation.json', record)
        return record
    except BaseException as error:
        record.update(status='FAILED', error=str(error), error_type=type(error).__name__,
            elapsed_seconds=time.perf_counter() - started, artifacts=artifacts, traceback=traceback.format_exc())
        _write_json(output / 'evaluation.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('acquisition', 'fields', 'training', 'config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--inference-migration', type=Path)
    parser.add_argument('--prior-refit', type=Path, help='Verified fitting-only AR prior cache for this selected network')
    parser.add_argument('--conditional-prior', type=Path, help='Separate fitting-only template/time controller mean variant')
    parser.add_argument('--continue-from', type=Path, help='Closed resource-stopped parent with a verified committed forecast prefix')
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        result = evaluate(args.acquisition, args.fields, args.training, args.config, args.output,
            args.device, inference_migration=args.inference_migration, prior_refit=args.prior_refit,
            conditional_prior=args.conditional_prior, continue_from=args.continue_from)
        print(json.dumps({key: value for key, value in result.items() if key != 'artifacts'}))
    except Exception as error:
        if not preexisting:
            args.output.mkdir(parents=True, exist_ok=True)
            _write_json(args.output / 'failure.json', dict(status='FAILED', error=str(error),
                error_type=type(error).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

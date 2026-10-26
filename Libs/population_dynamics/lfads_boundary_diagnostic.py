"""Cross LFADS boundary inference with future controller conditioning, without fitting.

Only prefix_boundary_prior_future is a forecast. The remaining arms condition on
future responses and are development diagnostics, including an explicit fixed
latent-input intervention. Full-response variational inference is approximate.
The source graph, its random sites, weights and original training lengths remain
unchanged. Full response encoders use the native joint-sample batch, preserving
the source's floating-point reduction order before caching their small outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml

from Libs.population_dynamics.lfads import (
    LFADSConditionalSimulator, LFADSCountSamples, LFADSRandomStreams, SOURCE_REVISION,
)
from Libs.population_dynamics.lfads_trajectory_evaluation import (
    _native_episode, _project_counts, _sample_payload, _verified_training,
)
from Libs.population_dynamics.lfads_trajectory_training import _write_json
from Libs.population_dynamics.poisson_trajectory_evaluation import fitting_scale
from Libs.population_dynamics.provenance import sha256_file
from Libs.population_dynamics import trajectory_sample_metrics as unit_metrics_module
from Libs.population_dynamics.trajectory_sample_metrics import unit_count_metrics
from Libs.population_dynamics.trajectory_scores import energy_score, field_pairs, variogram_score


BRANCHES = (
    'prefix_boundary_prior_future',
    'full_response_boundary_prior_future',
    'prefix_boundary_posterior_future',
    'full_response_boundary_posterior_future',
)
PROTOCOL = dict(
    schema='neural_population_lfads_boundary_diagnostic_v1',
    scope='development_on_shared_epoch_selection_blocks',
    training_seeds=[41, 42, 43], selection_blocks=[14, 15, 16, 17],
    expected_cases_per_seed=16, expected_units=100000,
    original_history_bins=200, original_horizon_bins=1000,
    forecast_horizon_bins=100, bin_ms=5.0, joint_samples=16, sampling_seed_base=86000,
    full_response_encoder_batch='joint_samples',
    branches=list(BRANCHES), horizons_ms=[20.0, 80.0, 500.0],
    temporal_lags_bins=[1, 4, 16], refractory_support_windows_bins=[1, 2, 4],
    endpoint_rtol=0.000005, endpoint_atol=0.0000001,
    formal_evaluation=False, update_weights=False,
    retain_full_unit_samples='first_selection_case_per_seed_all_four_branches',
)


def branch_role(branch: str, requested_role: str | None = None) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError('UNKNOWN_LFADS_BOUNDARY_DIAGNOSTIC_BRANCH')
    forecast = branch == BRANCHES[0]
    role = 'forecast' if forecast else 'response_conditioned_diagnostic'
    if requested_role is not None and requested_role != role:
        raise ValueError('NONCAUSAL_BRANCH_CANNOT_BE_CALLED_A_FORECAST_OR_INTERVENTION_SUCCESS')
    return dict(branch=branch, role=role, uses_future_responses=not forecast,
        boundary_conditioning=('observed_prefix' if branch.startswith('prefix_')
                               else 'complete_response_variational_smoothing'),
        future_controller_conditioning=('learned_AR_prior' if branch.endswith('_prior_future')
                                        else 'fixed_complete_response_variational_path'),
        latent_input_intervention=branch == BRANCHES[2],
        mechanistic_microstate_oracle=False, inverse_intervention_evidence=False,
        scientific_claim_qualified=False)


@dataclass
class PreparedBoundaryCrossing:
    """Small latent paths and source feeds; never retains replicated responses."""
    model: LFADSConditionalSimulator
    prefix_state: Any
    future_stimulus: np.ndarray
    sampling_seeds: tuple[int, int, int]
    n_samples: int
    bins: int
    prefix_feed: dict[Any, Any]
    full_response_feed: dict[Any, Any]
    full_response_native_feed: dict[Any, Any]
    initial: dict[str, np.ndarray]
    history: dict[str, np.ndarray]
    boundary: dict[str, np.ndarray]
    prior_future: dict[str, np.ndarray]
    posterior_future: np.ndarray
    initial_standard_noise: np.ndarray
    history_standard_noise: np.ndarray
    future_standard_noise: np.ndarray
    costs: dict[str, float]


def _finite(name: str, *values: np.ndarray) -> None:
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError('NONFINITE_LFADS_BOUNDARY_DIAGNOSTIC:' + name)


def _paired_noise(model, draws, bins, seeds):
    streams = LFADSRandomStreams.from_seeds(*seeds)
    initial = streams.initial.standard_normal((draws, 1, model.model_config['ic_dim'])).astype(np.float32)
    shape = (draws, model.model_config['co_dim'])
    history = np.stack([streams.initial.standard_normal(shape).astype(np.float32)
                        for _ in range(model.history_bins)], axis=1)
    future = np.stack([streams.dynamics.standard_normal(shape).astype(np.float32)
                       for _ in range(bins)], axis=1)
    return initial, history, future


class LFADSBoundaryDiagnostic:
    """Execute four conditioning arms through the unchanged source generator."""

    def __init__(self, model: LFADSConditionalSimulator):
        if (model.history_bins != model.training_history_bins
                or model.horizon_bins != model.training_horizon_bins):
            raise ValueError('ORIGINAL_LFADS_TRAINING_GRAPH_LENGTHS_REQUIRED')
        self.model = model

    def prepare(self, counts, stimulus, *, horizon_bins: int, n_samples: int,
                sampling_seeds: tuple[int, int, int]) -> PreparedBoundaryCrossing:
        model, started = self.model, time.perf_counter()
        total, h = model.history_bins + model.horizon_bins, model.history_bins
        if (type(horizon_bins) is not int or not 1 <= horizon_bins <= model.horizon_bins
                or type(n_samples) is not int or n_samples < 1):
            raise ValueError('POSITIVE_BOUNDED_DIAGNOSTIC_HORIZON_AND_DRAWS_REQUIRED')
        if np.shape(counts) != (1, total, len(model.unit_ids)):
            raise ValueError('ONE_COMPLETE_IDENTIFIED_RESPONSE_EPISODE_REQUIRED')
        streams = LFADSRandomStreams.from_seeds(*sampling_seeds)
        y = model._counts(counts, total)
        u = model._stimulus(stimulus, 1, total)
        state = model.infer_state(dict(counts=y[:, :h], stimulus=u[:, :h]))
        prefix_feed, prefix_latent, bins = model._draw_feed(
            state, u[:, h:h + horizon_bins], streams, n_samples)
        m, session = model._full.model, model._full.session
        prefix_boundary = session.run(m.gen_states[h - 1], feed_dict=prefix_feed)
        costs = dict(prefix_encoding_and_latent_sampling_seconds=time.perf_counter() - started)
        initial_noise, history_noise, future_noise = _paired_noise(model, n_samples, bins, sampling_seeds)

        # Match the native posterior's batch arithmetic. Encoding at batch one
        # changed near-zero future states beyond the frozen tolerance at100k
        # units. Broadcast the observed episode for this one encoder call, then
        # retain only the small encoder outputs, not the repeated responses.
        t = time.perf_counter()
        fetch = dict(mean=m.posterior_zs_g0.mean, logvar=m.posterior_zs_g0.logvar,
                     forward=m.ci_enc_fwd)
        if not model.model_config['do_causal_controller']:
            fetch['reverse'] = m.ci_enc_rev
        encoded = session.run(fetch, feed_dict=m.build_feed_dict('recording',
            np.broadcast_to(y, (n_samples, total, len(model.unit_ids))),
            np.broadcast_to(u, (n_samples, total, model.input_size)), keep_prob=1.))
        del y
        full_feed = {m.dataName: 'recording', m.batch_size: n_samples, m.keep_prob: 1.,
                     m.posterior_zs_g0.mean: encoded['mean'],
                     m.posterior_zs_g0.logvar: encoded['logvar'],
                     m.posterior_zs_g0.noise_bxn: initial_noise[:, 0]}
        if m.ext_input is not None:
            full_feed[m.ext_input] = np.broadcast_to(u, (n_samples, total, model.input_size))
        for name, tensors in (('forward', m.ci_enc_fwd), ('reverse', m.ci_enc_rev)):
            if name not in encoded:
                continue
            for tensor, value in zip(tensors, encoded[name]):
                _finite('complete_response_encoder', value)
                full_feed[tensor] = value
        _finite('complete_response_initial_distribution', encoded['mean'], encoded['logvar'])
        for tick in range(h + bins):
            full_feed[m.posterior_zs_co[tick].noise_bxn] = (
                history_noise[:, tick] if tick < h else future_noise[:, tick - h])
        costs['full_response_encoder_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        posteriors = m.posterior_zs_co[:h + bins]
        full = session.run(dict(initial=m.g0s_val, boundary=m.gen_states[h - 1],
            controller=[p.sample for p in posteriors], means=[p.mean for p in posteriors],
            logvars=[p.logvar for p in posteriors]), feed_dict=full_feed)
        _finite('complete_response_latent_path', full['initial'], full['boundary'],
                np.asarray(full['controller']), np.asarray(full['means']), np.asarray(full['logvars']))
        scales = np.exp(.5 * np.asarray(full['logvars']))
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise FloatingPointError('INVALID_COMPLETE_RESPONSE_CONTROLLER_VARIANCE')
        controller = np.stack(full['controller'], axis=1)
        native_feed = dict(full_feed)  # No imposed boundary or controller draws.
        full_feed[m.g0s_val] = full['initial']
        for tick in range(h):
            full_feed[m.controller_outputs[tick]] = controller[:, tick]
        phi, logvar = session.run([m.prior_zs_ar_con.phis_bxu,
                                  m.prior_zs_ar_con.logevars_1xu], feed_dict=full_feed)
        innovation_std = np.exp(.5 * logvar)
        if (not np.isfinite(phi).all() or (phi < 0).any() or (phi > 1).any()
                or not np.isfinite(innovation_std).all() or (innovation_std <= 0).any()):
            raise FloatingPointError('INVALID_FUTURE_AR_PRIOR')
        previous, full_prior = controller[:, h - 1], []
        for tick in range(bins):
            previous = (phi * previous + innovation_std * future_noise[:, tick]).astype(np.float32)
            full_prior.append(previous)
        costs['full_response_latent_sampling_seconds'] = time.perf_counter() - t
        return PreparedBoundaryCrossing(model, state, u[:, h:h + bins], sampling_seeds,
            n_samples, bins, prefix_feed, full_feed, native_feed,
            initial=dict(prefix=prefix_latent['initial_latent_states'],
                         full_response=full['initial'][:, None]),
            history=dict(prefix=prefix_latent['historical_controller_inputs'],
                         full_response=controller[:, None, :h]),
            boundary=dict(prefix=prefix_boundary[:, None], full_response=full['boundary'][:, None]),
            prior_future=dict(prefix=prefix_latent['future_controller_inputs'],
                              full_response=np.stack(full_prior, axis=1)[:, None]),
            posterior_future=controller[:, None, h:h + bins],
            initial_standard_noise=initial_noise, history_standard_noise=history_noise,
            future_standard_noise=future_noise, costs=costs)

    def _branch_feed(self, prepared, branch):
        branch_role(branch)
        if prepared.model is not self.model:
            raise ValueError('PREPARED_BOUNDARY_BELONGS_TO_ANOTHER_MODEL')
        boundary = 'prefix' if branch.startswith('prefix_') else 'full_response'
        feed = dict(prepared.prefix_feed if boundary == 'prefix' else prepared.full_response_feed)
        future = (prepared.prior_future[boundary] if branch.endswith('_prior_future')
                  else prepared.posterior_future)
        m, h = self.model._full.model, self.model.history_bins
        feed[m.gen_states[h - 1]] = prepared.boundary[boundary][:, 0]
        for tick in range(prepared.bins):
            feed[m.controller_outputs[h + tick]] = future[:, 0, tick]
        return feed, boundary, future

    def sample_arm(self, prepared: PreparedBoundaryCrossing, branch: str,
                   requested_role: str | None = None) -> LFADSCountSamples:
        branch_role(branch, requested_role)
        feed, boundary, controller = self._branch_feed(prepared, branch)
        values = self.model._read_future(feed, prepared.n_samples, prepared.bins)
        rng = LFADSRandomStreams.from_seeds(*prepared.sampling_seeds)
        counts = rng.observation.poisson(values['conditional_expected_counts'])
        return LFADSCountSamples(counts=counts, **values,
            initial_latent_states=prepared.initial[boundary],
            historical_controller_inputs=prepared.history[boundary],
            future_controller_inputs=controller, unit_ids=self.model.unit_ids,
            bin_seconds=self.model.bin_seconds)

    def verify_endpoint(self, prepared, branch, sample, *, rtol, atol):
        """Replay uncrossed endpoints; source posterior reference fixes no z/G.

        Raw-response versus cached-encoder equivalence is tested separately.
        Reference observations are generated one draw at a time in the public
        sampler's contiguous draw/batch/time/unit RNG order to bound peak memory.
        """
        if branch not in (BRANCHES[0], BRANCHES[3]):
            raise ValueError('ENDPOINT_CHECK_REQUIRES_AN_UNCROSSED_BRANCH')
        if branch == BRANCHES[0]:
            feed, _, bins = self.model._draw_feed(prepared.prefix_state, prepared.future_stimulus,
                LFADSRandomStreams.from_seeds(*prepared.sampling_seeds), prepared.n_samples)
        else:
            feed, bins = prepared.full_response_native_feed, prepared.bins
        reference = self.model._read_future(feed, prepared.n_samples, bins)
        details = dict(reference=('public_prefix_adapter' if branch == BRANCHES[0]
                                   else 'native_full_response_posterior_with_cached_encoder'),
                       rtol=rtol, atol=atol, arrays={}, all_within_tolerance=True)
        for name in ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states'):
            left, right = getattr(sample, name), reference[name]
            exact, close, max_abs, max_scaled = True, True, 0., 0.
            for a, b in zip(left, right):
                exact = exact and np.array_equal(a, b)
                difference = np.abs(a.astype(np.float64) - b)
                denominator = atol + rtol * np.abs(b.astype(np.float64))
                max_abs = max(max_abs, float(difference.max()))
                max_scaled = max(max_scaled, float(np.max(difference / denominator)))
                close = close and bool(np.all(difference <= denominator))
            details['arrays'][name] = dict(bitwise_equal=bool(exact), within_tolerance=bool(close),
                                          max_abs_error=max_abs, maximum_tolerance_fraction=max_scaled)
            if not close:
                raise ValueError('LFADS_BOUNDARY_ENDPOINT_MISMATCH:' + branch + ':' + name)
        observation = np.random.default_rng(prepared.sampling_seeds[2])
        equal = True
        for draw, means in enumerate(reference['conditional_expected_counts']):
            equal = equal and np.array_equal(observation.poisson(means), sample.counts[draw])
        details['sampled_counts_bitwise_equal'] = bool(equal)
        # Exact public replay is the hard causal contract. Native posterior output
        # comparisons use the frozen floating tolerance and report count equality.
        if branch == BRANCHES[0] and (not equal or any(
                not row['bitwise_equal'] for row in details['arrays'].values())):
            raise ValueError('CAUSAL_BOUNDARY_ENDPOINT_REQUIRES_EXACT_PUBLIC_REPLAY')
        return details


def _load_protocol(path):
    config = yaml.safe_load(Path(path).read_text())
    if config != PROTOCOL:
        raise ValueError('EXACT_FROZEN_LFADS_BOUNDARY_DIAGNOSTIC_PROTOCOL_REQUIRED')
    return config


def evaluate(acquisition, fields, training, config_path, output, device_name='cuda', *,
             inference_migration=None, causal_reference=None):
    started = time.perf_counter()
    acquisition, fields, training, config_path, output = map(Path,
        (acquisition, fields, training, config_path, output))
    if output.exists():
        raise FileExistsError('LFADS boundary diagnostic output already exists')
    config = _load_protocol(config_path)
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('EXPLICIT_CPU_OR_CUDA_REQUIRED')
    if device_name == 'cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise ValueError('QUALIFIED_LFADS_CUDA_REQUIRES_NVIDIA_TF32_OVERRIDE_0')
    cohort, run, directory, selected, migration = _verified_training(
        acquisition, fields, training, inference_migration)
    binding, fit_config = run['binding'], run['binding']['config']
    if (binding['seed'] not in config['training_seeds']
            or fit_config['validation_blocks'] != config['selection_blocks']
            or len(cohort.validation) != config['expected_cases_per_seed']
            or len(cohort.unit_ids) != config['expected_units']
            or fit_config['history_bins'] != config['original_history_bins']
            or fit_config['horizon_bins'] != config['original_horizon_bins']
            or fit_config['bin_ms'] != config['bin_ms']):
        raise ValueError('DIAGNOSTIC_TRAINING_GRAPH_COHORT_OR_SEED_CHANGED')
    blocks = {case['block'] for case in cohort.validation}
    templates = {case['template_id'] for case in cohort.validation}
    if (blocks != set(config['selection_blocks']) or len(templates) != 4
            or {(case['block'], case['template_id']) for case in cohort.validation}
               != {(block, template) for block in blocks for template in templates}):
        raise ValueError('ALL_FOUR_SELECTION_BLOCKS_AND_FOUR_TEMPLATES_REQUIRED')
    native = _native_episode(cohort, cohort.validation[0])
    intervals, source = native.minimum_interval_ticks.copy(), native.source_identity
    scale_mean, scale_sd, metadata = fitting_scale(fields, cohort.fit)
    if (not np.array_equal(native.field_ids, metadata['field_ids'])
            or not np.array_equal(np.bincount(native.unit_field_index, minlength=len(native.field_ids)),
                                  metadata['unit_count'])):
        raise ValueError('DIAGNOSTIC_FIELD_IDENTITIES_OR_DENOMINATORS_CHANGED')
    definitions = [json.loads(str(value)) for value in metadata['field_ids']]
    pairs = field_pairs([r[0] for r in definitions], [r[2] for r in definitions], [r[1] for r in definitions])
    torch.set_num_threads(fit_config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    h, bins, draws = config['original_history_bins'], config['forecast_horizon_bins'], config['joint_samples']
    horizons = [int(ms / config['bin_ms']) for ms in config['horizons_ms']]
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    record = dict(schema='lfads_boundary_conditioning_diagnostic_evaluation_v1', status='RUNNING',
        model_type='lfads_boundary_diagnostic', configuration=config, configuration_sha256=sha256_file(config_path),
        training_seed=binding['seed'], training_binding=binding,
        training_run_sha256=sha256_file(training / 'run.json'), selected_checkpoint=run['best_checkpoint'],
        selected_training_sha256=sha256_file(directory / 'training.json'),
        selected_model_artifacts=selected['model_artifacts'], selected_epoch=run['best_epoch'],
        source_revision=SOURCE_REVISION, native_source_identity=source, inference_migration=migration,
        dataset_sha256=binding['dataset_sha256'], acquisition_design_sha256=binding['acquisition_design_sha256'],
        evaluator_implementation_sha256=sha256_file(Path(__file__)),
        unit_metrics_implementation_sha256=sha256_file(Path(unit_metrics_module.__file__)),
        branch_roles={branch: branch_role(branch) for branch in BRANCHES},
        all_unit_outputs_generated=True, field_outputs_from_unit_counts=True,
        formal_test_accessed=False, calibration_outcomes_accessed=False,
        scientific_claim_qualified=False, inverse_intervention_evidence=False,
        mechanistic_microstate_oracle=False, weights_updated=False,
        posterior_interpretation='source variational approximation, not an exact posterior or true microstate',
        paired_noise='same standardized IC/history/future normals and observation seeds across arms',
        inference='descriptive epoch-selection diagnostics only; no significance or equivalence claim',
        unit_metric_estimator='shared float32 CPU Torch', field_metric_estimator='shared float64 fitting normalized',
        boundary='generator hidden state and previous controller input after bin199',
        full_response_encoder_batch=draws, original_graph_bins=h + config['original_horizon_bins'],
        cost_scope='shared preparation plus per-arm full-unit generation, exact integer projection, scoring and serialization',
        peak_tensorflow_gpu_allocated_bytes=None, host_memory_measurement='process-lifetime Linux ru_maxrss',
        completed_cases=0, completed_branches=0, score_rows=0, endpoint_checks=[], case_bindings=[],
        causal_reference=None, artifacts=artifacts)
    _write_json(output / 'evaluation.json', record)
    np.savez(output / 'fitting_field_scale.npz', mean_hz=scale_mean, standard_deviation_hz=scale_sd, **metadata)
    artifacts['fitting_field_scale.npz'] = sha256_file(output / 'fitting_field_scale.npz')
    del native
    try:
        graph_started = time.perf_counter()
        with LFADSConditionalSimulator.load(directory, device=device_name) as model:
            cohort.verify_model(model)
            diagnostic = LFADSBoundaryDiagnostic(model)
            record['graph_setup_seconds'] = time.perf_counter() - graph_started
            record['execution_device'] = model.execution_device
            for case_index, case in enumerate(cohort.validation):
                case_started = time.perf_counter()
                data = _native_episode(cohort, case, source, intervals)
                count, stimulus = data.window(0, data.duration_ticks)
                if count.shape != (h + config['original_horizon_bins'], len(cohort.unit_ids)):
                    raise ValueError('COMPLETE_ORIGINAL_NATIVE_RESPONSE_REQUIRED')
                load_seconds = time.perf_counter() - case_started
                seed = config['sampling_seed_base'] + binding['seed'] * 100000 + case_index * 100
                prepared = diagnostic.prepare(count[None], stimulus[None], horizon_bins=bins,
                    n_samples=draws, sampling_seeds=(seed, seed + 1, seed + 2))
                record['case_bindings'].append(dict(case_id=case['case_id'], binding=data.binding,
                    sampling_seeds=list(prepared.sampling_seeds), preparation_costs=prepared.costs))
                observed = count[h:h + bins]
                for branch in BRANCHES:
                    branch_started = time.perf_counter()
                    sample = diagnostic.sample_arm(prepared, branch)
                    generation_seconds = time.perf_counter() - branch_started
                    conditional = [float(sample.conditional_expected_counts[:, :, :length].sum(
                        dtype=np.float64) / (draws * len(cohort.unit_ids))) for length in horizons]
                    endpoint_seconds = 0.
                    if branch in (BRANCHES[0], BRANCHES[3]):
                        t = time.perf_counter()
                        check = diagnostic.verify_endpoint(prepared, branch, sample,
                            rtol=config['endpoint_rtol'], atol=config['endpoint_atol'])
                        record['endpoint_checks'].append(dict(case_id=case['case_id'], branch=branch, **check))
                        endpoint_seconds = time.perf_counter() - t
                    paths = _sample_payload(sample, cohort.unit_ids, draws, h, bins,
                                            config['bin_ms'] / 1000, fit_config['model'])
                    generated = sample.counts[:, 0]
                    del sample
                    if causal_reference is not None and case_index == 0 and branch == BRANCHES[0]:
                        with np.load(causal_reference, allow_pickle=False) as previous:
                            if (not np.array_equal(previous['unit_ids'], cohort.unit_ids)
                                    or not np.array_equal(previous['sampled_unit_counts'], generated)):
                                raise ValueError('ARCHIVED_CAUSAL_FIRST_CASE_FULL_UNIT_REPLAY_MISMATCH')
                        record['causal_reference'] = dict(path=str(causal_reference),
                            sha256=sha256_file(causal_reference), matched_count_values=int(generated.size))
                    t = time.perf_counter()
                    raw_fields, raw_truth = _project_counts(generated, observed,
                        cohort.unit_field_index, len(cohort.field_ids))
                    projection_seconds = time.perf_counter() - t
                    sampled = (raw_fields / (metadata['unit_count'] * config['bin_ms'] / 1000) - scale_mean) / scale_sd
                    truth = (raw_truth / (metadata['unit_count'] * config['bin_ms'] / 1000) - scale_mean) / scale_sd
                    name = f"{case['case_id']}_{branch}.npz"
                    t = time.perf_counter()
                    np.savez_compressed(output / name, observed_field_counts=raw_truth,
                        sampled_field_counts=raw_fields, unit_ids=cohort.unit_ids,
                        field_ids=cohort.field_ids, unit_field_index=cohort.unit_field_index,
                        sampling_seeds=np.asarray(prepared.sampling_seeds, dtype=np.int64),
                        known_input_sequence=stimulus, **paths)
                    artifacts[name] = sha256_file(output / name)
                    if case_index == 0:
                        full_name = 'full_unit_' + branch + '.npz'
                        np.savez_compressed(output / full_name, sampled_unit_counts=generated,
                                            unit_ids=cohort.unit_ids)
                        artifacts[full_name] = sha256_file(output / full_name)
                    serialization_seconds = time.perf_counter() - t
                    t = time.perf_counter()
                    generated_tensor = torch.from_numpy(generated)
                    observed_tensor = torch.from_numpy(observed).to(dtype=torch.float32)
                    rows = []
                    for index, (ms, length) in enumerate(zip(config['horizons_ms'], horizons)):
                        row = dict(seed=binding['seed'], block=case['block'], case_id=case['case_id'],
                            template_id=case['template_id'], origin_ms=h * config['bin_ms'], horizon_ms=ms,
                            method='lfads', **branch_role(branch),
                            conditional_expected_spikes_per_unit=conditional[index],
                            energy_score=energy_score(sampled[:, :length], truth[:length]),
                            **unit_count_metrics(generated_tensor[:, :length], observed_tensor[:length],
                                intervals, data.bin_ticks, config['refractory_support_windows_bins']),
                            full_unit_generation_seconds=generation_seconds,
                            input_loading_seconds=load_seconds, **prepared.costs,
                            endpoint_check_seconds=endpoint_seconds, field_projection_seconds=projection_seconds,
                            sample_serialization_seconds=serialization_seconds,
                            graph_setup_seconds=record['graph_setup_seconds'], ensemble_artifact=name)
                        for label, pair in pairs.items():
                            row['variogram_' + label] = variogram_score(sampled[:, :length], truth[:length],
                                                                        pair[:, 0], pair[:, 1])
                        for lag in config['temporal_lags_bins']:
                            row[f'variogram_temporal_lag_{lag}'] = (variogram_score(sampled[:, :length], truth[:length],
                                np.arange(len(cohort.field_ids)), np.arange(len(cohort.field_ids)), lag)
                                if lag < length else None)
                        rows.append(row)
                    scoring_seconds = time.perf_counter() - t
                    for row in rows:
                        row.update(scoring_seconds=scoring_seconds,
                            branch_elapsed_seconds=time.perf_counter() - branch_started,
                            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                    with (output / 'scores.csv').open('a', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        if record['score_rows'] == 0:
                            writer.writeheader()
                        writer.writerows(rows)
                    record['score_rows'] += len(rows)
                    record['completed_branches'] += 1
                    _write_json(output / 'evaluation.json', record)
                    print(json.dumps(dict(status='DIAGNOSING_DEVELOPMENT', case_id=case['case_id'],
                        branch=branch, training_seed=binding['seed'], rows=record['score_rows'])), flush=True)
                    del generated, generated_tensor, observed_tensor, paths, raw_fields, sampled, rows
                record['completed_cases'] += 1
                _write_json(output / 'evaluation.json', record)
                del prepared, count, stimulus, data, observed
        expected = config['expected_cases_per_seed'] * len(BRANCHES)
        if (record['completed_cases'] != config['expected_cases_per_seed']
                or record['completed_branches'] != expected
                or record['score_rows'] != expected * len(horizons)
                or len(record['endpoint_checks']) != 2 * config['expected_cases_per_seed']):
            raise ValueError('LFADS_BOUNDARY_DIAGNOSTIC_PROTOCOL_INCOMPLETE')
        record.update(status='DIAGNOSED_DEVELOPMENT', elapsed_seconds=time.perf_counter() - started,
                      scores_sha256=sha256_file(output / 'scores.csv'))
        _write_json(output / 'evaluation.json', record)
        return record
    except BaseException as error:
        record.update(status='FAILED', error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        _write_json(output / 'evaluation.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('acquisition', 'fields', 'training', 'config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--inference-migration', type=Path)
    parser.add_argument('--causal-reference', type=Path)
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        record = evaluate(args.acquisition, args.fields, args.training, args.config,
            args.output, args.device, inference_migration=args.inference_migration,
            causal_reference=args.causal_reference)
        print(json.dumps(dict(status=record['status'], training_seed=record['training_seed'],
                              completed_cases=record['completed_cases'], score_rows=record['score_rows'])))
    except Exception as error:
        if not preexisting:
            args.output.mkdir(parents=True, exist_ok=True)
            _write_json(args.output / 'failure.json', dict(status='FAILED', error=str(error),
                error_type=type(error).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

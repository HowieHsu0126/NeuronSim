"""Exact first-origin counts and opt-in independent all-origin unit statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_renewal_data import NativeRenewalEpisode, load_dataset_index
from .poisson_state_space import LatentRandomStreams
from .poisson_trajectory_training import partition
from .population_trajectory_evaluation import (
    _apply_correction, _draw, _initial_memory_arrays, _load_model, sample_population_outputs,
)
from .population_trajectory_evaluation import COST_POINT_UNITS, COST_POINT_FIELDS
from .provenance import sha256_file
from .renewal_check import replay_events
from .trajectory_scores import energy_score, field_pairs, variogram_score


# Frozen before all-origin metric comparisons; the evaluator reduces float32.
METRIC_RTOL = 5e-6
METRIC_ATOL = 1e-7
UNIT_CHUNK_SIZE = 2048
ALL_ORIGIN_HORIZONS_MS = (20., 80., 500.)
UNIT_METRICS = ("unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit",
                "observed_spikes_per_unit", "unit_total_coverage_90",
                "unit_total_interval_width_90")


def _linear_quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def numpy_unit_metrics(samples, observed, intervals, bin_ticks, horizons, support_windows,
                       *, chunk_units=UNIT_CHUNK_SIZE):
    """Independent float64 moments, order statistics and integer support counts.

    Every unit and draw is consumed; finite-draw corrected MSE may be negative.
    Chunking bounds working arrays without selecting a subset of units.
    """
    samples, observed, intervals = map(np.asarray, (samples, observed, intervals))
    if (samples.ndim != 3 or min(samples.shape) < 1 or len(samples) < 2
            or samples.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or observed.shape != samples.shape[1:] or observed.dtype.kind not in 'iuf'
            or intervals.shape != (samples.shape[-1],) or intervals.dtype.kind not in 'iu'
            or np.any(intervals < 1) or type(bin_ticks) is not int or bin_ticks < 1
            or type(chunk_units) is not int or chunk_units < 1):
        raise ValueError("identified integer samples, observations and source intervals required")
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or not 1 <= h <= samples.shape[1] for h in horizons)
            or max(horizons) != samples.shape[1]
            or len(set(support_windows)) != len(support_windows)
            or any(type(w) is not int or w < 1 for w in support_windows)):
        raise ValueError("unique complete horizons and positive support windows required")
    draws, bins, units = samples.shape
    sums = {h: dict(mse=0., expected=0., observed=0., coverage=0, width=0.,
                    violations={w: 0 for w in support_windows}) for h in horizons}
    consumed = 0
    for start in range(0, units, chunk_units):
        stop = min(start + chunk_units, units)
        integer, target = samples[:, :, start:stop], observed[:, start:stop]
        # Per-unit horizon totals must remain exact both in int64 and float64.
        maximum_count = (2**53 - 1) // bins
        if (np.any(integer < 0) or np.any(integer > maximum_count)
                or not np.isfinite(target).all() or np.any(target < 0)
                or np.any(target != np.floor(target)) or np.any(target > maximum_count)):
            raise ValueError("finite nonnegative exactly representable count totals required")
        values, truth = integer.astype(np.float64), target.astype(np.float64)
        cumulative = np.empty((draws, bins + 1, stop - start), dtype=np.int64)
        cumulative[:, 0] = 0
        np.cumsum(integer, axis=1, dtype=np.int64, out=cumulative[:, 1:])
        consumed += integer.size
        for horizon in horizons:
            current, actual = values[:, :horizon], truth[:horizon]
            mean = current.mean(axis=0)
            variance = np.square(current - mean[None]).sum(axis=0) / (draws - 1)
            totals = cumulative[:, horizon].astype(np.float64)
            actual_total = actual.sum(axis=0)
            ordered = np.sort(totals, axis=0)
            lower, upper = _linear_quantile(ordered, .05), _linear_quantile(ordered, .95)
            row = sums[horizon]
            row['mse'] += float(np.sum(np.square(mean - actual) - variance / draws))
            row['expected'] += float(totals.sum())
            row['observed'] += float(actual_total.sum())
            row['coverage'] += int(np.count_nonzero((actual_total >= lower) & (actual_total <= upper)))
            row['width'] += float((upper - lower).sum())
            for width in support_windows:
                if width <= horizon:
                    maximum = (width * bin_ticks - 1) // intervals[start:stop] + 1
                    rolling = cumulative[:, width:horizon + 1] - cumulative[:, :horizon + 1 - width]
                    row['violations'][width] += int(np.count_nonzero(rolling > maximum))
    metrics, exact = {}, {}
    for horizon in horizons:
        row = sums[horizon]
        metrics[horizon] = dict(unit_bin_mean_mse_corrected=row['mse'] / (horizon * units),
            sampled_expected_spikes_per_unit=row['expected'] / (draws * units),
            observed_spikes_per_unit=row['observed'] / units,
            unit_total_coverage_90=row['coverage'] / units,
            unit_total_interval_width_90=row['width'] / units)
        exact[horizon] = {'unit_total_coverage_90': (row['coverage'], units)}
        for width in support_windows:
            name = f'refractory_count_violation_{width}bin'
            numerator = row['violations'][width] if width <= horizon else None
            denominator = draws * (horizon - width + 1) * units if width <= horizon else None
            metrics[horizon][name] = numerator / denominator if denominator is not None else None
            exact[horizon][name] = (numerator, denominator)
    if consumed != samples.size:
        raise ValueError("incomplete unit-count metric denominator")
    return metrics, exact, consumed


def _project_units(sampled, observed, mapping, fields):
    """Independent NumPy projection, including every identified unit and truth."""
    if (mapping.shape != (sampled.shape[-1],) or mapping.dtype.kind not in 'iu'
            or np.any(mapping < 0) or np.any(mapping >= fields)
            or observed.shape != sampled.shape[1:]):
        raise ValueError("unit-to-field projection identities differ")
    generated = np.zeros((*sampled.shape[:2], fields), dtype=np.int64)
    truth = np.zeros((sampled.shape[1], fields), dtype=np.int64)
    for start in range(0, len(mapping), UNIT_CHUNK_SIZE):
        stop = min(start + UNIT_CHUNK_SIZE, len(mapping))
        local = mapping[start:stop]
        for field in np.unique(local):
            selected = local == field
            generated[:, :, field] += sampled[:, :, start:stop][:, :, selected].sum(-1, dtype=np.int64)
            truth[:, field] += observed[:, start:stop][:, selected].sum(-1, dtype=np.int64)
    return generated, truth


def _exact_array(saved, actual, label):
    if saved.dtype != actual.dtype or not np.array_equal(saved, actual):
        raise ValueError(label)


def _score_rows(forecasts, evaluation, cases):
    cfg = evaluation['configuration']
    origins, horizons = cfg['origins_ms'], cfg['horizons_ms']
    if (tuple(horizons) != ALL_ORIGIN_HORIZONS_MS or not origins or len(set(origins)) != len(origins)
            or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in origins)
            or cfg['finite_ensemble_corrections'] is not True or cfg['unit_metric_dtype'] != 'float32'
            or cfg['formal_evaluation'] is not False
            or cfg['retain_full_unit_samples'] != 'first_validation_case_and_first_origin_per_seed_and_model'):
        raise ValueError("declared all-origin 20/80/500ms unit-metric protocol required")
    names = (*UNIT_METRICS, *(f'refractory_count_violation_{w}bin' for w in cfg['refractory_support_windows_bins']))
    expected = {(case['case_id'], float(origin), float(h)): case
                for case in cases for origin in origins for h in horizons}
    expected_artifacts = {'full_unit_samples.npz', *(f"{c['case_id']}_origin_{o:g}.npz" for c in cases for o in origins)}
    if (evaluation['episodes'] != len(cases) or evaluation['score_rows'] != len(expected)
            or set(evaluation['artifacts']) != expected_artifacts):
        raise ValueError("complete declared episode/origin/artifact denominator required")
    with (forecasts / 'scores.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'seed', 'block', 'case_id', 'template_id', 'origin_ms', 'horizon_ms', 'method', 'ensemble_artifact', *names}
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not required <= set(reader.fieldnames)):
            raise ValueError("unique identified unit-metric CSV columns required")
        rows = list(reader)
    indexed = {}
    for row in rows:
        key = (row['case_id'], float(row['origin_ms']), float(row['horizon_ms']))
        if key not in expected or key in indexed or None in row:
            raise ValueError("missing, duplicate or unexpected forecast score rows")
        case = expected[key]
        if (int(row['seed']) != evaluation['training_seed'] or int(row['block']) != case['block']
                or row['method'] != evaluation.get('method', evaluation['model_type']) or row['template_id'] != case['template_id']
                or row['ensemble_artifact'] != f"{case['case_id']}_origin_{key[1]:g}.npz"):
            raise ValueError("forecast score row identity differs")
        indexed[key] = row
    if set(indexed) != set(expected) or len(rows) != evaluation['score_rows']:
        raise ValueError("complete forecast score rows required")
    return indexed


def _metric_comparisons(metrics, exact, row, seeds, scope):
    result = []
    for name, value in metrics.items():
        text = row[name]
        if value is None:
            if text != '':
                raise ValueError("unsupported-window metric must be empty: " + name)
            reported = absolute = allowed = None
            passed = True
        else:
            reported = float(text)
            if not np.isfinite(value) or not np.isfinite(reported):
                raise ValueError("nonfinite unit metric: " + name)
            absolute, allowed = abs(value - reported), METRIC_ATOL + METRIC_RTOL * abs(reported)
            passed = absolute <= allowed
        numerator, denominator = exact.get(name, (None, None))
        result.append(dict(model=row['method'], training_seed=int(row['seed']), case_id=row['case_id'],
            block=int(row['block']), template_id=row['template_id'], origin_ms=float(row['origin_ms']),
            horizon_ms=float(row['horizon_ms']), **seeds, unit_count_reference=scope,
            metric=name, evaluator_float32=reported, numpy_float64=value,
            absolute_difference=absolute, allowed_difference=allowed, passed=passed,
            exact_numerator=numerator, exact_denominator=denominator))
    return result


@torch.no_grad()
def replay(acquisition, fields, training, forecasts, output, device_name="cuda", *, all_origins=False,
           correction_path=None):
    evaluation = json.loads((forecasts / "evaluation.json").read_text())
    corrected = "correction_identity" in evaluation
    source_statuses = ("EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", "EVALUATED_V1_DEVELOPMENT_CHECKPOINT")
    if (correction_path is not None or corrected) and (evaluation.get("status") in source_statuses
                                                      or evaluation.get("model_type") != "population"):
        raise ValueError("residual correction requires the native population replay route")
    if corrected != (correction_path is not None):
        raise ValueError("replay requires exactly the recorded correction artifact")
    if corrected and not isinstance(evaluation["correction_identity"], dict):
        raise ValueError("recorded correction identity must be an object")
    if evaluation.get("status") == "EVALUATED_PFC_DEVELOPMENT_CHECKPOINT":
        if fields is not None:
            raise ValueError("PFC_REPLAY_USES_DECLARED_CIRCUIT_INPUTS")
        from .pfc_population_evaluation import replay as replay_pfc
        return replay_pfc(acquisition,training,forecasts,output,device_name)
    if evaluation.get("status") == "EVALUATED_V1_DEVELOPMENT_CHECKPOINT":
        if fields is not None:
            raise ValueError("V1_REPLAY_USES_DECLARED_CIRCUIT_INPUTS")
        from .v1_population_evaluation import replay as replay_v1
        return replay_v1(acquisition,training,forecasts,output,device_name)
    output = Path(output)
    csv_path = output.with_suffix('.csv')
    if output.exists() or (all_origins and csv_path.exists()):
        raise FileExistsError("existing replay evidence will not be overwritten")
    if all_origins and output.suffix != '.json':
        raise ValueError("all-origin replay output must be a new JSON path with a new CSV sibling")
    if evaluation["status"] != "EVALUATED_DEVELOPMENT" or evaluation['model_type'] not in ('poisson', 'renewal', 'population'):
        raise ValueError("complete matched evaluation required")
    if not corrected and evaluation.get("method", evaluation.get("model_type")) != evaluation.get("model_type"):
        raise ValueError("uncorrected replay method identity differs")
    for name, digest in evaluation["artifacts"].items():
        if Path(name).name != name or sha256_file(forecasts / name) != digest:
            raise ValueError("forecast artifact changed")
    if sha256_file(forecasts / "scores.csv") != evaluation["scores_sha256"]:
        raise ValueError("forecast metrics changed")
    cfg, model_type = evaluation['configuration'], evaluation['model_type']
    run = json.loads((training / 'run.json').read_text())
    design = json.loads((acquisition / 'design.json').read_text())
    _, cases = partition(design, run['binding']['config'])
    if not cases or len({case['case_id'] for case in design['cases']}) != len(design['cases']):
        raise ValueError("unique complete validation case identities required")
    indexed = _score_rows(forecasts, evaluation, cases) if all_origins else None
    index = load_dataset_index(fields)
    if set(index) != {case['case_id'] for case in design['cases']}:
        raise ValueError("replay cohort differs from canonical field index")
    for case_id, row in index.items():
        if (sha256_file(acquisition / case_id / 'report.json') != row['raw_report_sha256']
                or sha256_file(fields / case_id / 'summary.json') != row['field_summary_sha256']):
            raise ValueError("replay metadata differ from canonical field index")
    native = NativeRenewalEpisode(acquisition / cases[0]['case_id'], fields / cases[0]['case_id'])
    device = torch.device(device_name)
    model, binding, checkpoint = _load_model(training, acquisition, fields, model_type, device, native)
    if (checkpoint != evaluation['checkpoint_sha256'] or binding != run['binding']
            or evaluation['training_seed'] != binding['seed']
            or any(evaluation[key] != binding[key] for key in ('dataset_sha256', 'acquisition_design_sha256'))
            or native.source_identity != evaluation['native_source_identity']):
        raise ValueError("replay model/data/source differs from the evaluated binding")
    if all_origins or corrected:
        for key, name in [('evaluator_implementation_sha256', 'population_trajectory_evaluation.py'),
                          ('unit_metrics_implementation_sha256', 'trajectory_sample_metrics.py')]:
            if evaluation[key] != sha256_file(Path(__file__).with_name(name)):
                raise ValueError("recorded evaluation implementation changed")
        if evaluation['evaluation_blocks'] != binding['config']['validation_blocks']:
            raise ValueError("evaluation block identity differs")
    if corrected:
        model = _apply_correction(model, binding, checkpoint, correction_path)
        if (model.correction_identity != evaluation["correction_identity"]
                or evaluation.get("method") != "population_residual_" + model.mode):
            raise ValueError("replay correction artifact or method identity differs")
    reference = dict(unit_ids=native.unit_ids.copy(), mapping=native.unit_field_index.copy(),
        field_ids=native.field_ids.copy(), intervals=native.minimum_interval_ticks.copy(),
        source=native.source_identity, bin_ms=native.bin_ms, bin_ticks=native.bin_ticks)
    if binding['config']['bin_ms'] != native.bin_ms:
        raise ValueError("replay clock differs from the fitted observation map")
    horizons = [int(round(ms / native.bin_ms)) for ms in cfg['horizons_ms']]
    if any(h < 1 or h * native.bin_ms != ms for h, ms in zip(horizons, cfg['horizons_ms'])):
        raise ValueError("positive aligned horizons required")
    longest, draws = max(horizons), cfg['joint_samples']
    torch.set_num_threads(binding['config']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    comparisons, origin_records, first_report = [], [], None
    started = time.perf_counter()
    for case_index, case in enumerate(cases if all_origins else cases[:1]):
        data = native if case_index == 0 else NativeRenewalEpisode(
            acquisition / case['case_id'], fields / case['case_id'], expected_source=reference['source'],
            expected_unit_ids=reference['unit_ids'], expected_unit_field_index=reference['mapping'])
        canonical = index[case['case_id']]
        if (data.binding['episode_report_sha256'] != canonical['raw_report_sha256']
                or data.binding['field_summary_sha256'] != canonical['field_summary_sha256']
                or data.source_identity != reference['source'] or data.bin_ms != reference['bin_ms']
                or data.bin_ticks != reference['bin_ticks']
                or any(not np.array_equal(actual, reference[key]) for actual, key in
                    [(data.unit_ids, 'unit_ids'), (data.unit_field_index, 'mapping'),
                     (data.field_ids, 'field_ids'), (data.minimum_interval_ticks, 'intervals')])):
            raise ValueError("replay episode identity, observation map or source interval differs")
        count, stimulus = data.window(0, data.duration_ticks)
        y, u = torch.as_tensor(count, device=device), torch.as_tensor(stimulus, device=device)
        for origin_index, origin in enumerate(cfg['origins_ms'] if all_origins else cfg['origins_ms'][:1]):
            boundary = int(round(origin / data.bin_ms))
            if boundary < 1 or boundary * data.bin_ms != origin or boundary + longest > len(y):
                raise ValueError("complete aligned causal prefix and future support required")
            ready = data.ready_at(boundary * data.bin_ticks)
            history = dict(counts=y[None, :boundary], stimulus=u[None, :boundary])
            if model_type in ('renewal', 'population'):
                history['ready_in_ticks'] = ready[None]
            state = model.infer_state(history)
            future = u[None, boundary:boundary + longest]
            seed_base = cfg['sampling_seed_base'] + binding['seed'] * 100000 + case_index * 100 + origin_index * 3
            seeds = dict(initial_seed=seed_base, dynamics_seed=seed_base + 1, observation_seed=seed_base + 2)
            counts, latent, initial, terminal = _draw(model, model_type, state, future, seed_base, draws, device)
            actual = dict(sampled_unit_counts=counts.cpu().numpy(), unit_ids=data.unit_ids,
                initial_ready_in_ticks=ready, latent_states=latent.cpu().numpy(),
                initial_latent_states=initial.cpu().numpy())
            memory_arrays = _initial_memory_arrays(state, corrected)
            actual.update(memory_arrays)
            if terminal is not None:
                actual['terminal_ready_in_ticks'] = terminal.cpu().numpy()
            first = case_index == 0 and origin_index == 0
            scope = 'EXACT_SAVED_UNIT_COUNTS' if first else 'REGENERATED_COUNTS_WITH_METRIC_AND_FIELD_LATENT_CHECKS'
            if first:
                with np.load(forecasts / 'full_unit_samples.npz', allow_pickle=False) as saved:
                    if set(saved.files) != set(actual):
                        raise ValueError("full-unit replay schema differs")
                    for key, value in actual.items():
                        _exact_array(saved[key], value, 'full-unit elementwise replay differs: ' + key)
            observed = count[boundary:boundary + longest]
            metric_values = exact = None
            if all_origins:
                metric_values, exact, consumed = numpy_unit_metrics(actual['sampled_unit_counts'], observed,
                    reference['intervals'], data.bin_ticks, horizons, cfg['refractory_support_windows_bins'])
            projected, observed_field = _project_units(actual['sampled_unit_counts'], observed,
                                                      data.unit_field_index, len(data.field_ids))
            name = f"{case['case_id']}_origin_{origin:g}.npz"
            with np.load(forecasts / name, allow_pickle=False) as saved:
                expected = dict(sampled_field_counts=projected, observed_field_counts=observed_field,
                    latent_states=actual['latent_states'], initial_latent_states=actual['initial_latent_states'], **memory_arrays)
                if set(saved.files) != set(expected):
                    raise ValueError("field/latent replay schema differs")
                for key, value in expected.items():
                    _exact_array(saved[key], value, 'independent field/latent replay differs: ' + key)
            support = extra_seconds = None
            if first and model_type in ('renewal', 'population'):
                event_started = time.perf_counter()
                streams = LatentRandomStreams.from_seeds(
                    seed_base, seed_base + 1, seed_base + 2, device=device)
                if model_type == 'population':
                    detailed_counts, _, _, _, detailed_events = sample_population_outputs(
                        model, state, future, streams, draws, record_events=True)
                else:
                    detailed = model.simulate(state, future, streams, draws, record_events=True)
                    if not detailed.events_recorded:
                        raise ValueError("renewal event output missing")
                    detailed_counts, detailed_events = detailed.counts, detailed.events
                if not torch.equal(detailed_counts[:, 0], counts):
                    raise ValueError("event recording changes the sampled count trajectories")
                support, terminal_from_events = replay_events(actual['sampled_unit_counts'][:, None],
                    detailed_events.cpu().numpy(), ready[None], data.minimum_interval_ticks, data.bin_ticks)
                if not np.array_equal(terminal_from_events, actual['terminal_ready_in_ticks']):
                    raise ValueError("event-derived terminal availability differs")
                extra_seconds = time.perf_counter() - event_started
                del detailed_counts, detailed_events
                if model_type == "renewal":
                    del detailed
            if first:
                first_report = dict(status='EXACT_FULL_UNIT_AND_INDEPENDENT_FIELD_REPLAY', model_type=model_type,
                    training_seed=binding['seed'], case_id=case['case_id'], origin_ms=origin,
                    unit_count_values=counts.numel(), unit_ids=len(data.unit_ids), field_values=projected.size,
                    latent_values=latent.numel(), source_checkpoint_sha256=checkpoint,
                    full_unit_artifact_sha256=evaluation['artifacts']['full_unit_samples.npz'],
                    event_support=support, additional_event_validation_seconds=extra_seconds,
                    event_validation_excluded_from_matched_count_timing=True, scientific_fidelity_qualified=False)
            if all_origins:
                for ms, horizon in zip(cfg['horizons_ms'], horizons):
                    row = indexed[(case['case_id'], float(origin), float(ms))]
                    comparisons.extend(_metric_comparisons(metric_values[horizon], exact[horizon], row, seeds, scope))
                origin_records.append(dict(case_id=case['case_id'], case_index=case_index, origin_ms=origin,
                    origin_index=origin_index, **seeds, ensemble_artifact=name,
                    ensemble_artifact_sha256=evaluation['artifacts'][name], unit_count_reference=scope,
                    sampled_unit_count_values=consumed, observed_unit_count_values=observed.size,
                    field_count_values=projected.size + observed_field.size,
                    latent_values=actual['latent_states'].size + actual['initial_latent_states'].size,
                    fields_and_latents_exact=True, native_episode_binding=data.binding))
                print(json.dumps(dict(status='REPLAYED_ORIGIN', model=model_type, seed=binding['seed'],
                    case_id=case['case_id'], origin_ms=origin, completed_origins=len(origin_records))), flush=True)
            # Release views before loading another complete native episode.
            del actual, expected, counts, latent, initial, terminal, state, projected, observed_field, memory_arrays
            del history, future, observed
        del y, u, count, stimulus, data
        native = None
    report = first_report
    if all_origins:
        passed = all(row['passed'] for row in comparisons)
        finite = [row for row in comparisons if row['absolute_difference'] is not None]
        report = dict(status='PASS' if passed else 'MISMATCH', model_type=model_type, training_seed=binding['seed'],
            verified_origins=len(origin_records), expected_origins=len(cases) * len(cfg['origins_ms']),
            verified_score_rows=len(indexed), metric_comparisons=len(comparisons),
            all_forecast_unit_metrics_replayed=passed, all_origin_fields_and_latents_exact=True,
            saved_full_unit_origins_compared=1, regenerated_origins_without_saved_full_unit_arrays=len(origin_records) - 1,
            all_origin_unit_counts_compared_to_saved_full_arrays=False,
            sampled_unit_count_values_consumed=sum(r['sampled_unit_count_values'] for r in origin_records),
            horizons_ms=cfg['horizons_ms'], first_origin_exact_replay=first_report, origins=origin_records,
            sampling_seed_derivation='sampling_seed_base + training_seed*100000 + case_index*100 + origin_index*3; dynamics +1; observation +2',
            sampling_seed_base=cfg['sampling_seed_base'], case_order='partition: lexicographically sorted case_id',
            origin_order='evaluation.configuration.origins_ms',
            metric_tolerance=dict(rtol=METRIC_RTOL, atol=METRIC_ATOL), tolerance_predeclared=True,
            metric_computation='independent chunked NumPy float64; centered variance ddof1; linear order-statistic quantiles; integer rolling support counts',
            metric_units='MSE counts squared per unit/bin; totals and interval widths are counts, not Hz',
            maximum_absolute_difference=max(r['absolute_difference'] for r in finite),
            maximum_fraction_of_allowed_difference=max(r['absolute_difference'] / r['allowed_difference'] for r in finite),
            source_checkpoint_sha256=checkpoint, evaluation_sha256=sha256_file(forecasts / 'evaluation.json'),
            scores_sha256=evaluation['scores_sha256'], dataset_sha256=binding['dataset_sha256'],
            acquisition_design_sha256=binding['acquisition_design_sha256'],
            unit_metrics_implementation_sha256=evaluation['unit_metrics_implementation_sha256'],
            replay_implementation_sha256=sha256_file(Path(__file__)), elapsed_seconds=time.perf_counter() - started,
            field_distribution_scores_recomputed=False, scientific_fidelity_qualified=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader(); writer.writerows(comparisons)
        report.update(csv_artifact=csv_path.name, csv_sha256=sha256_file(csv_path))
    if corrected:
        report.update(method=evaluation["method"], correction_identity=model.correction_identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if all_origins and report['status'] != 'PASS':
        raise ValueError("predeclared unit-metric comparison failed; retained immutable JSON/CSV deltas")
    return report


@torch.no_grad()
def replay_cost_point(acquisition, fields, training, delivery, output, device_name="cuda"):
    """Replay the complete registered cost ensemble and independent unit metrics."""
    acquisition, fields, training, delivery, output = map(Path, (acquisition, fields, training, delivery, output))
    if output.exists() or output.resolve().is_relative_to(delivery.resolve()):
        raise ValueError("new replay output outside the immutable cost delivery required")
    started = time.perf_counter()
    report = dict(status="RUNNING", simulation_performed=False, training_performed=False,
                  scientific_claim_qualified=False, matched_fidelity_qualified=False, speedup_qualified=False,
                  delivery_sha256=sha256_file(delivery / "delivery.json"))
    try:
        proof = json.loads((delivery / "delivery.json").read_text())
        if (proof.get("status") != "PASS" or proof.get("exit_code") != 0
                or proof.get("generated_unit_count_values") != 16 * 100 * COST_POINT_UNITS):
            raise ValueError("successful full-unit cost delivery required")
        archive_rows = proof["files"]
        if len({row["path"] for row in archive_rows}) != len(archive_rows):
            raise ValueError("unique cost archive paths required")
        for row in archive_rows:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("relative cost archive path required")
            path = delivery / relative
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError("cost archive differs: " + str(relative))
        root = delivery / "cost_point"
        evaluation = json.loads((root / "cost_point.json").read_text())
        if (evaluation["status"] != "EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT"
                or evaluation["all_unit_outputs_saved"] is not True or evaluation["draws"] != 16
                or evaluation["score_rows"] != 3 or evaluation["generated_bins"] != 100
                or evaluation["unit_count"] != COST_POINT_UNITS or evaluation["field_count"] != COST_POINT_FIELDS
                or evaluation["generated_unit_count_values"] != 16 * 100 * COST_POINT_UNITS):
            raise ValueError("complete registered cost point required")
        device = torch.device(device_name)
        data = NativeRenewalEpisode(acquisition / evaluation["configuration"]["case_id"],
                                    fields / evaluation["configuration"]["case_id"])
        if (len(data.unit_ids) != COST_POINT_UNITS or len(data.field_ids) != COST_POINT_FIELDS
                or data.dt_ms != .1 or data.bin_ms != 5. or data.duration_ticks * data.dt_ms != 6000.
                or evaluation["boundary_tick"] != 10000 or evaluation["observed_history_bins"] != 200):
            raise ValueError("complete source scale and cost-point observation boundary required")
        model, binding, checkpoint = _load_model(training, acquisition, fields, "population", device, data)
        if (checkpoint != evaluation["checkpoint_sha256"] or checkpoint != proof["binding"]["checkpoint_sha256"]
                or data.binding != evaluation["source_binding"] or binding["seed"] != 41
                or data.source_identity != evaluation["native_source_identity"]):
            raise ValueError("original checkpoint or native source differs")
        torch.set_num_threads(binding["config"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        boundary, bins = evaluation["boundary_tick"], evaluation["observed_history_bins"]
        counts, controls = data.window(0, boundary + 100 * data.bin_ticks)
        y, u = torch.as_tensor(counts, device=device), torch.as_tensor(controls, device=device)
        ready = data.ready_at(boundary)
        state = model.infer_state(dict(counts=y[None, :bins], stimulus=u[None, :bins], ready_in_ticks=ready[None]))
        seeds = evaluation["sampling_seeds"]
        if seeds != dict(initial=89000, dynamics=89001, observation=89002):
            raise ValueError("registered new cost-point random streams required")
        generated, latent, initial, terminal = _draw(model, "population", state, u[None, bins:bins + 100], 89000, 16, device)
        samples = np.load(root / "sampled_unit_counts.npy", mmap_mode="r", allow_pickle=False)
        _exact_array(samples, generated.cpu().numpy(), "cost-point generated counts differ")
        with np.load(root / "ensemble_fields.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        for name, actual in (("unit_ids", data.unit_ids), ("unit_field_index", data.unit_field_index),
                             ("field_ids", data.field_ids), ("initial_ready_in_ticks", ready),
                             ("latent_states", latent.cpu().numpy()), ("initial_latent_states", initial.cpu().numpy()),
                             ("terminal_ready_in_ticks", terminal.cpu().numpy()),
                             ("stimulus", u[None, bins:bins + 100].cpu().numpy())):
            _exact_array(arrays[name], actual, "cost-point array differs: " + name)
        observed = y[bins:bins + 100].cpu().numpy().astype(np.int64)
        _exact_array(arrays["observed_unit_counts"], observed, "cost-point observed counts differ")
        projected, truth_fields = _project_units(samples, observed, data.unit_field_index, len(data.field_ids))
        _exact_array(arrays["sampled_field_counts"], projected, "cost-point sampled fields differ")
        _exact_array(arrays["observed_field_counts"], truth_fields, "cost-point truth fields differ")
        scale_path = delivery / "forecasts/fitting_field_scale.npz"
        if sha256_file(scale_path) != evaluation["fitting_scale_sha256"]:
            raise ValueError("cost-point fitting scale differs")
        with np.load(scale_path, allow_pickle=False) as scale:
            denominators = scale["unit_count"]
            if not np.array_equal(denominators, np.bincount(data.unit_field_index, minlength=len(data.field_ids))):
                raise ValueError("cost-point field denominator differs")
            scaled = (projected / (denominators * data.bin_ms / 1000) - scale["mean_hz"]) / scale["standard_deviation_hz"]
            truth = (truth_fields / (denominators * data.bin_ms / 1000) - scale["mean_hz"]) / scale["standard_deviation_hz"]
        definitions = [json.loads(str(name)) for name in data.field_ids]
        pairs = field_pairs([row[0] for row in definitions], [row[2] for row in definitions], [row[1] for row in definitions])
        metrics, exact, consumed = numpy_unit_metrics(samples, observed, data.minimum_interval_ticks, data.bin_ticks,
                                                      [4, 16, 100], [1, 2, 4])
        with (root / "scores.csv").open(newline="") as stream:
            scores = list(csv.DictReader(stream))
        if len(scores) != 3 or [float(row["horizon_ms"]) for row in scores] != [20., 80., 500.]:
            raise ValueError("all registered cost-point horizons required")
        checked = 0
        for row, horizon in zip(scores, [4, 16, 100]):
            for name, expected in metrics[horizon].items():
                text = row[name]
                if (text != "" if expected is None else
                        not np.isfinite(float(text)) or abs(float(text) - expected) > METRIC_ATOL + METRIC_RTOL * abs(float(text))):
                    raise ValueError("independent cost-point unit metric differs: " + name)
                checked += 1
            expected_fields = dict(energy_score=energy_score(scaled[:, :horizon], truth[:horizon]))
            for name, pair in pairs.items():
                expected_fields["variogram_" + name] = variogram_score(scaled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
            for lag in [1, 4, 16]:
                expected_fields[f"variogram_temporal_lag_{lag}"] = (variogram_score(scaled[:, :horizon], truth[:horizon],
                    np.arange(len(data.field_ids)), np.arange(len(data.field_ids)), lag) if lag < horizon else None)
            for name, expected in expected_fields.items():
                if (row[name] != "" if expected is None else float(row[name]) != expected):
                    raise ValueError("cost-point field metric differs: " + name)
                checked += 1
        report.update(status="PASS", exact_generated_unit_count_values=int(samples.size),
                      independently_consumed_unit_count_values=consumed, checked_score_cells=checked,
                      archive_files_verified=len(archive_rows), archive_bytes_verified=sum(row["bytes"] for row in archive_rows),
                      exact_counts_latents_readiness_and_fields=True, checkpoint_sha256=checkpoint,
                      replay_device=str(device), unit_metric_rtol=METRIC_RTOL, unit_metric_atol=METRIC_ATOL,
                      process_seconds=proof["process_seconds"], task_through_archive_seconds=proof["task_through_archive_seconds"])
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["replay_seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition', 'training', 'forecasts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--all-origins', action='store_true')
    parser.add_argument('--cost-point', action='store_true', help='Replay one archived full-unit cost ensemble')
    parser.add_argument('--correction', type=Path, help='Authenticated native population residual correction directory')
    a = parser.parse_args()
    if (a.fields is None and json.loads((a.forecasts / 'evaluation.json').read_text()).get('status')
            not in ('EVALUATED_PFC_DEVELOPMENT_CHECKPOINT', 'EVALUATED_V1_DEVELOPMENT_CHECKPOINT')):
        parser.error('--fields is required for native Mesocircuit replay')
    if a.cost_point:
        if a.correction is not None:
            parser.error('--correction is not supported for cost-point replay')
        if a.all_origins:
            parser.error("cost-point and all-origins are separate replay scopes")
        print(json.dumps(replay_cost_point(a.acquisition, a.fields, a.training, a.forecasts, a.output, a.device)))
        return
    print(json.dumps(replay(a.acquisition, a.fields, a.training, a.forecasts, a.output,
                            a.device, all_origins=a.all_origins, correction_path=a.correction)))


if __name__ == '__main__':
    main()

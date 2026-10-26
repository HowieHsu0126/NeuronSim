"""Describe response scales and endpoint domains before PFC confirmation design.

This consumes completed development readouts. It does not select error margins,
release reserved blocks, or convert empirical bootstrap spread into confidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .pfc_delay_analysis import PHASES, read, require, write_json, write_rows
from .pfc_deterministic_replay import _constants, SOURCE_ENTRY_SHA256
from .pfc_phase_responses import CONDITIONS, METRICS, CONTRASTS
from .provenance import sha256_file
from .pfc_scale_analysis import _source_manifest, _source_bound_file

PRECISION_TABLES = {
    'field_mean': 'field_mean_sampling_precision.csv',
    'field_lag_covariance': 'field_lag_covariance_sampling_precision.csv',
    'count_cdf': 'count_cdf_sampling_precision.csv',
    'source_response': 'source_response_sampling_precision.csv',
    'paired_response': 'paired_response_sampling_precision.csv',
}
CLOCK_LABELS = ('numpy_dt_0p05', 'numpy_dt_0p025', 'numpy_dt_0p0125')


def bound_file(root, report, name):
    path = root/name
    binding = report['outputs'][name]
    require(path.stat().st_size == binding['bytes'] and sha256_file(path) == binding['sha256'],
            'calibration output binding changed: '+name)
    return path


def load_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def maximum_count(window_ms, dt_ms, refractory_ms):
    """At most 1+floor((W-1)/R) spikes in W integer ticks, R ticks apart.

    The source refractory gate is checked at every tick. Windows include their
    first tick and exclude their endpoint; no extra spike is added at that end.
    """
    values = (window_ms, dt_ms, refractory_ms)
    require(all(np.isfinite(x) and x > 0 for x in values), 'positive finite clock and intervals required')
    window, refractory = round(window_ms/dt_ms), round(refractory_ms/dt_ms)
    require(window > 0 and refractory > 0
            and np.isclose(window*dt_ms, window_ms, rtol=0, atol=1e-9)
            and np.isclose(refractory*dt_ms, refractory_ms, rtol=0, atol=1e-9),
            'intervals must be integral source ticks')
    return 1+(window-1)//refractory


def endpoint_domain(family, row, dt_ms, refractory_ms):
    """Separate structural identities, missing lag pairs and empirical spread."""
    original = row['status']
    physical_maximum = None
    structural = None
    if family == 'field_lag_covariance':
        start, stop = PHASES[row['phase']]
        lag = float(row['lag_ms'])
        if lag >= stop-start:
            require(original == 'UNDEFINED_PHASE_CONTAINED_LAG' and row['estimate'] == '',
                    'undefined phase-contained lag must remain unestimated')
            return 'UNDEFINED_NO_WITHIN_PHASE_PAIRS', 'lag_not_shorter_than_phase', None
    require(original in ('ESTIMATED_EMPIRICAL_SPREAD', 'EMPIRICALLY_DEGENERATE'),
            'unexpected endpoint status outside undefined lag')
    if family == 'count_cdf':
        field = int(row['field_index'])
        require(0 <= field < 17, 'source field index outside 16E+1I map')
        population = 'E' if field < 16 else 'I'
        physical_maximum = maximum_count(float(row['window_ms']), dt_ms, refractory_ms[population])
        if int(row['count_threshold']) >= physical_maximum:
            structural = 'cdf_is_one_above_source_refractory_support'
    elif family == 'paired_response' and row['phase'] == 'baseline':
        structural = 'identical_pre_cue_inputs_and_initial_state'
    if structural is not None:
        require(float(row['estimate']) == 0.0 and float(row['bootstrap_standard_error']) == 0.0,
                'structural zero contradicts archived estimates')
        return 'STRUCTURAL_ZERO_CONTRAST', structural, physical_maximum
    return ('EMPIRICALLY_DEGENERATE_UNCERTAIN' if original == 'EMPIRICALLY_DEGENERATE'
            else 'ESTIMATED_EMPIRICAL_SPREAD'), 'no_population_uncertainty_claim', physical_maximum


def response_tables(cohort, values, seeds, config):
    x = np.asarray(values, dtype=float)
    require(x.shape == (len(CONDITIONS), len(seeds), len(PHASES), len(METRICS))
            and len(seeds) >= 2 and len(set(seeds)) == len(seeds) and np.isfinite(x).all(),
            'complete finite condition/block/phase/response array required')
    require(np.array_equal(x[0, :, 0], x[1, :, 0]) and np.array_equal(x[0, :, 0], x[2, :, 0]),
            'paired pre-cue source responses differ')
    blocks, summaries, paired = [], [], []
    base = dict(cohort=cohort, coupled_units=config['excitatory_neurons']+config['inhibitory_neurons'],
                dt_ms=config['dt_ms'], joint_rng_blocks=len(seeds), units='Hz')
    def summary(v):
        return dict(mean=float(v.mean()), block_sd=float(v.std(ddof=1)),
                    block_minimum=float(v.min()), block_maximum=float(v.max()),
                    empirical_zero_blocks=int(np.count_nonzero(v == 0.)),
                    uncertainty='descriptive_block_scatter_not_confidence_interval')
    for ci, condition in enumerate(CONDITIONS):
        for pi, (phase, (start, stop)) in enumerate(PHASES.items()):
            for mi, metric in enumerate(METRICS):
                key = dict(base, condition=condition, phase=phase, start_ms=start, stop_ms=stop, metric=metric)
                v = x[ci, :, pi, mi]
                summaries.append(dict(key, **summary(v)))
                for seed, value in zip(seeds, v):
                    blocks.append(dict(key, source_rng_seed=seed, value=float(value)))
    for contrast, positive, negative in CONTRASTS:
        delta = x[CONDITIONS.index(positive)]-x[CONDITIONS.index(negative)]
        for pi, (phase, (start, stop)) in enumerate(PHASES.items()):
            for mi, metric in enumerate(METRICS):
                v = delta[:, pi, mi]
                paired.append(dict(base, contrast=contrast, phase=phase, start_ms=start, stop_ms=stop,
                                   metric=metric, **summary(v)))
    return blocks, summaries, paired


def native_responses(root, delivery_hash):
    require(sha256_file(root/'delivery.json') == delivery_hash, 'native phase delivery changed')
    delivery = read(root/'delivery.json')
    manifest = _source_manifest(delivery)
    report = read(_source_bound_file(root, manifest, 'analysis/analysis.json'))
    replay_path = _source_bound_file(root, manifest, 'analysis/readout_replay.json')
    rows_path = _source_bound_file(root, manifest, 'analysis/responses.csv')
    require(report['status'] == 'DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC'
            and report['units'] == 2560 and report['source_rng_seeds'] == [93001, 93002, 93003]
            and report['complete_source_episodes'] == 9 and report['phases_ms'] == PHASES
            and report['readout_replay_sha256'] == sha256_file(replay_path)
            and report['artifacts']['responses.csv'] == sha256_file(rows_path),
            'complete previously verified native phase cohort required')
    replay = read(replay_path)
    require(replay['status'] == 'PASS' and replay['artifacts'] == report['artifacts'],
            'native response replay binding differs')
    config, seeds = report['configuration'], report['source_rng_seeds']
    expected = {(f'pfc_{seed}_{condition}', phase) for seed in seeds for condition in CONDITIONS for phase in PHASES}
    rows = load_rows(rows_path)
    require(len(rows) == len(expected) and {(r['case_id'], r['phase']) for r in rows} == expected,
            'native phase identities incomplete or duplicated')
    indexed = {(r['case_id'], r['phase']): r for r in rows}
    values = np.empty((3, len(seeds), len(PHASES), len(METRICS)))
    for ci, condition in enumerate(CONDITIONS):
        for bi, seed in enumerate(seeds):
            for pi, (phase, (start, stop)) in enumerate(PHASES.items()):
                row = indexed[f'pfc_{seed}_{condition}', phase]
                require(int(row['source_rng_seed']) == seed and row['condition'] == condition
                        and float(row['start_ms']) == start and float(row['stop_ms']) == stop
                        and int(row['units']) == 2560, 'native response metadata changed')
                scalar = {name: float(row[name]) for name in METRICS if not name.startswith('vector_')}
                for axis in ('x', 'y'):
                    scalar['vector_'+axis+'_hz'] = float(row['vector_'+axis+'_spikes'])/(config['excitatory_neurons']*(stop-start)/1000)
                values[ci, bi, pi] = [scalar[name] for name in METRICS]
    return values, seeds, config


def describe(config_path, project_root, output):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='RUNNING', new_source_simulations=0, production_episodes=0,
                  confirmation_execution_released=False, numerical_margins_frozen=False,
                  scientific_fidelity_qualified=False, confidence_intervals=False)
    try:
        settings = yaml.safe_load(config_path.read_text())
        require(settings['schema'] == 'pfc_calibration_design_inputs_v1'
                and settings['scope'] == 'development_calibration', 'development design inputs required')
        comparison = project_root/settings['comparison_archive']
        report_path = comparison/'analysis/report.json'
        require(sha256_file(report_path) == settings['comparison_report_sha256'], 'comparison report changed')
        report = read(report_path)
        require(report['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
                and report['source_episodes_reused'] == 297 and report['confidence_intervals'] is False,
                'complete fixed calibration comparison required')
        source_path = project_root/settings['source_entry']
        require(sha256_file(source_path) == SOURCE_ENTRY_SHA256, 'pinned source equations changed')
        refractory = {p: 1000*_constants({'weight_scaling_factor': 1.}, p)['refractory'] for p in ('E', 'I')}
        require(refractory == {'E': 2., 'I': 1.}, 'source refractory support changed')
        configs, origins = {}, {}
        for label, path in settings['fluctuation_archives'].items():
            root = project_root/path
            expected = report['inputs'][label]
            require(sha256_file(root/'delivery.json') == expected['delivery_sha256'], 'field source delivery changed')
            manifest = _source_manifest(read(root/'delivery.json'))
            configs[label] = yaml.safe_load(_source_bound_file(root, manifest, 'config.yaml').read_text())
            field_path = _source_bound_file(root, manifest, 'analysis/field_fluctuations.npz')
            require(configs[label]['dt_ms'] == expected['dt_ms']
                    and configs[label]['excitatory_neurons']+configs[label]['inhibitory_neurons'] == expected['coupled_units'],
                    'source dimensions or clock changed')
            origins[label] = dict(path=str(path), delivery_sha256=expected['delivery_sha256'], field_path=field_path)
        require(set(configs) == {*CLOCK_LABELS, 'native_2560'}, 'all four field cohorts required')
        response_path = bound_file(comparison/'analysis', report, 'block_source_responses.npz')
        blocks, summaries, paired = [], [], []
        with np.load(response_path, allow_pickle=False) as stored:
            seeds = stored['source_rng_seeds'].tolist()
            require(seeds == list(range(95001, 95033))
                    and stored['condition_names'].tolist() == list(CONDITIONS)
                    and stored['phase_names'].tolist() == list(PHASES)
                    and stored['metric_names'].tolist() == list(METRICS), 'calibration axes or reserved seeds changed')
            for label in CLOCK_LABELS:
                new = response_tables(label, stored[label+'_values'], seeds, configs[label])
                for destination, rows in zip((blocks, summaries, paired), new): destination.extend(rows)
        native_root = project_root/settings['native_phase_archive']
        native, native_seeds, native_config = native_responses(native_root, settings['native_phase_delivery_sha256'])
        require(all(native_config[k] == configs['native_2560'][k] for k in
                    ('dt_ms', 'excitatory_neurons', 'inhibitory_neurons', 'stimulus_center_deg', 'stimulus_width_deg')),
                'native phase/field configurations differ')
        for destination, rows in zip((blocks, summaries, paired), response_tables('native_2560', native, native_seeds, native_config)):
            destination.extend(rows)
        field_rows = []
        cov_path = bound_file(comparison/'analysis', report, 'lagged_field_covariance_hz2.npz')
        with np.load(cov_path, allow_pickle=False) as covariances:
            for label, origin in origins.items():
                expected = report['inputs'][label]
                with np.load(origin['field_path'], allow_pickle=False) as stored:
                    sizes = stored['field_unit_counts']
                    require(sizes.tolist() == expected['field_unit_counts']
                            and stored['source_rng_seeds'].tolist() == expected['source_rng_blocks'], 'field axes changed')
                    for condition in CONDITIONS:
                        counts = stored[condition+'_field_counts']
                        require(counts.shape == (len(expected['source_rng_blocks']), 1000, 17), 'field count coverage changed')
                        for phase, (start, stop) in PHASES.items():
                            mean = (counts[:, round(start/5):round(stop/5)]/(sizes*.005)).mean(axis=(0, 1))
                            covariance = covariances[f'{label}_{condition}_{phase}_lag_0ms']
                            variance = np.diag(covariance)
                            require(covariance.shape == (17, 17) and np.isfinite(variance).all() and (variance >= 0).all(),
                                    'invalid source marginal field variance')
                            for field in range(17):
                                field_rows.append(dict(cohort=label, condition=condition, phase=phase, field_index=field,
                                    population='E' if field < 16 else 'I', units_in_field=int(sizes[field]),
                                    joint_rng_blocks=len(expected['source_rng_blocks']), mean_rate_hz=float(mean[field]),
                                    matched_time_marginal_sd_hz=float(np.sqrt(variance[field])),
                                    single_event_bin_rate_quantum_hz=float(1/(sizes[field]*.005)),
                                    covariance_floor_chosen=False, error_margin_chosen=False))
        domains, support_groups = [], defaultdict(set)
        for family, name in PRECISION_TABLES.items():
            rows = load_rows(bound_file(comparison/'analysis', report, name))
            require(len(rows) == report['row_counts'][name], 'precision endpoint row count changed')
            for row in rows:
                dt = configs[row['left_cohort']]['dt_ms']
                domain, reason, maximum = endpoint_domain(family, row, dt, refractory)
                identity = {k: row[k] for k in ('left_cohort','right_cohort','condition','contrast','phase','field_index',
                            'leading_field','trailing_field','lag_ms','window_ms','count_threshold','metric') if k in row}
                domains.append(dict(family=family, endpoint_identity=json.dumps(identity, sort_keys=True),
                    empirical_status=row['status'], definition_status=domain, reason=reason,
                    physical_maximum_count=maximum, estimate=row['estimate']))
                if family == 'count_cdf':
                    key = tuple(row[k] for k in ('left_cohort','right_cohort','condition','phase','window_ms','field_index'))
                    support_groups[key].add(int(row['count_threshold']))
        require(len(domains) == report['calibration_precision']['diagnostics']['requested_endpoints'], 'complete endpoint accounting differs')
        require(len({(r['family'], r['endpoint_identity']) for r in domains}) == len(domains),
                'endpoint identities incomplete or duplicated')
        support = []
        for key, thresholds in support_groups.items():
            left, right, condition, phase, window, field = key
            population = 'E' if int(field) < 16 else 'I'
            maximum = maximum_count(float(window), configs[left]['dt_ms'], refractory[population])
            require(maximum == maximum_count(float(window), configs[right]['dt_ms'], refractory[population]),
                    'clock-specific support requires separate domains')
            missing = sorted(set(range(maximum))-thresholds)
            support.append(dict(left_cohort=left, right_cohort=right, condition=condition, phase=phase,
                window_ms=float(window), field_index=int(field), physical_maximum_count=maximum,
                archived_threshold_minimum=min(thresholds), archived_threshold_maximum=max(thresholds),
                unrepresented_nontrivial_thresholds=len(missing), first_unrepresented_threshold=missing[0] if missing else None,
                interpretation='unobserved_support_is_not_missing_simulation_data_or_zero_population_mass'))
        tables = {'response_block_values.csv': blocks, 'response_scales.csv': summaries,
                  'paired_response_scales.csv': paired, 'marginal_field_scales.csv': field_rows,
                  'endpoint_domains.csv': domains, 'count_support.csv': support}
        for name, rows in tables.items(): write_rows(output/name, rows)
        result.update(status='COMPLETE_DESCRIPTIVE_CONFIRMATION_DESIGN_INPUTS', settings=settings,
            settings_sha256=sha256_file(config_path), source_entry_sha256=SOURCE_ENTRY_SHA256,
            refractory_ms=refractory, source_episodes_reused=297, calibration_episodes=288,
            independent_blocks={'native_2560': native_seeds, **{label: seeds for label in CLOCK_LABELS}},
            independent_unit='joint_initial_background_rng_block', paired_across_sizes=False,
            source_coordinates='360*source_E_index/N_E; signed vectors retained for every block',
            covariance_scale='sqrt of existing across-block matched-time variance averaged over phase times',
            physical_support='right-open complete integer-tick window; spikes at least source refractory ticks apart',
            definition_status_counts=dict(Counter(row['definition_status'] for row in domains)),
            endpoint_statuses_by_family={family:dict(Counter(r['definition_status'] for r in domains if r['family']==family)) for family in PRECISION_TABLES},
            unrepresented_nontrivial_cdf_thresholds=sum(r['unrepresented_nontrivial_thresholds'] for r in support),
            all_seed_condition_failures_retained_in_upstream=True,
            field_scale_floor_is_not_error_margin=True, confirmation_replication_selected=False,
            interpretation='Development response and scale descriptions only. Empirical degeneracy remains uncertain except independently justified source identities. Count support accounting is not a confidence band. No size-equivalence or continuous-memory claim.',
            row_counts={name:len(rows) for name,rows in tables.items()},
            implementation_sha256=sha256_file(Path(__file__)),
            outputs={name:dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name)) for name in tables})
    except Exception as error:
        result.update(status='FAILED', error=str(error), error_type=type(error).__name__)
        write_json(output/'report.json', result)
        raise
    write_json(output/'report.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    describe(args.config, args.project_root, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

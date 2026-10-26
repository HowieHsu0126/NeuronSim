"""Descriptive PFC clock and size comparisons from verified fluctuation archives."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from .pfc_delay_analysis import PHASES, read, require, write_json, write_rows
from .pfc_scale_analysis import _source_manifest, _source_bound_file, _source_registration
from . import pfc_count_histograms, pfc_numerical_precision, pfc_phase_responses
from .pfc_fluctuation_analysis import unit_mapping
from .source_variant_analysis import time_inner_product_gram, covariance_from_gram
from .provenance import sha256_file

CONDITIONS = ("omission", "cue", "strong_cue")
CLOCKS = (.05, .025, .0125)
METRICS = ("mean_count_per_unit", "mean_single_unit_count_variance",
           "population_mean_count_variance", "diagonal_contribution_to_population_variance",
           "ordered_offdiagonal_contribution_to_population_variance", "pooled_single_unit_ensemble_fano")
PHYSICAL_KEYS = ("expected_source_revision", "duration_ms", "stimulus_center_deg", "stimulus_width_deg",
    "stimulus_start_ms", "stimulus_duration_ms", "distractor_center_deg", "distractor_width_deg",
    "distractor_start_ms", "distractor_duration_ms", "readout_bin_ms", "excitatory_preference_bins")


def lag_covariance(rates, lag):
    """Cov_B(leading field at t, trailing field at t+lag), averaged over t."""
    x = np.asarray(rates, dtype=np.float64)
    require(x.ndim == 3 and x.shape[0] >= 2 and min(x.shape[1:]) > 0
            and np.isfinite(x).all() and type(lag) is int and lag >= 0,
            "finite block/time/field rates and nonnegative integer lag required")
    if lag >= x.shape[1]:
        return None
    paired = x if lag == 0 else np.concatenate((x[:, :-lag], x[:, lag:]), axis=2)
    full = covariance_from_gram(time_inner_product_gram(paired),
                                np.ones((1, len(x)), dtype=np.int64))[0]
    return full if lag == 0 else full[:x.shape[2], x.shape[2]:]


def difference_summary(left, right):
    if left is None or right is None:
        return dict(status="UNDEFINED_PHASE_CONTAINED_LAG", mean_difference=None,
                    mae=None, rmse=None, maximum_absolute_difference=None)
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
            "matching finite comparison arrays required")
    delta = a-b
    return dict(status="DEFINED", mean_difference=float(delta.mean()),
                mae=float(np.abs(delta).mean()), rmse=float(np.sqrt(np.square(delta).mean())),
                maximum_absolute_difference=float(np.abs(delta).max()))


def load_cohort(root, native=False, *, registered_size=None):
    require(registered_size is None or (not native and type(registered_size) is int and registered_size in (2560, 100000)),
            "explicit streamed size role required")
    root = Path(root)
    delivery = read(root/"delivery.json")
    require(delivery.get("status") == "ARCHIVED_PFC_SOURCE_READOUTS" and delivery.get("exit_code") == 0,
            "completed delivered fluctuation archive required")
    require(delivery.get("readout_mode") == ("reference-fluctuations" if native else "fluctuations"),
            "fluctuation source role changed")
    manifest = _source_manifest(delivery)
    report = read(_source_bound_file(root, manifest, "analysis/report.json"))
    require(report.get("status") == "COMPLETE_SOURCE_FLUCTUATION_CALIBRATION"
            and all(report.get(key) is False for key in
                    ("confidence_intervals", "p_values", "q_values", "scientific_fidelity_qualified")),
            "complete descriptive fluctuation result required")
    config_path = _source_bound_file(root, manifest, "config.yaml")
    config = yaml.safe_load(config_path.read_text())
    require(sha256_file(config_path) == report["config_sha256"], "source configuration changed")
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    total = n_e+n_i
    seeds = [93001, 93002, 93003] if native else [94001, 94002, 94003]
    if registered_size is not None or (not native and "source_replication" in config):
        _, _, seeds, _ = _source_registration(config_path)
    require(report.get("completed_source_cases") == report.get("registered_source_cases") == 3*len(seeds),
            "complete declared fluctuation cohort required")
    size_valid = (total == registered_size if registered_size is not None
                  else ((n_e, n_i) == (2048, 512) if native else total >= 100000))
    require(total == report["coupled_core_units"] and size_valid
            and report["dt_ms"] == config["dt_ms"] and config["dt_ms"] in CLOCKS
            and (not native or config["dt_ms"] == .05)
            and report["joint_rng_blocks"] == seeds and report["phases_ms"] == PHASES
            and config["readout_bin_ms"] == report["bin_ms"] == 5.
            and config["duration_ms"] == 5000. and config["excitatory_preference_bins"] == 16
            and n_e*config["weight_scaling_factor"] == 2048
            and n_i*config["weight_scaling_factor"] == 512,
            "source clock, scale, phase or block identity changed")
    expected = {f"pfc_{seed}_{condition}": (seed, condition)
                for seed in seeds for condition in CONDITIONS}
    cases = config["cases"]
    require(len(cases) == len(expected) and {row["case_id"] for row in cases} == set(expected)
            and len(report["source_bindings"]) == len(expected)
            and {row["case_id"] for row in report["source_bindings"]} == set(expected),
            "all unique registered source cases required")
    strengths = dict(omission=0., cue=.06, strong_cue=.5)
    for row in cases:
        seed, condition = expected[row["case_id"]]
        require(row["source_rng_seed"] == seed and row["stimulus_strength_nA"] == strengths[condition]
                and row["distractor_strength_nA"] == 0., "source control registration changed")
    files = {}
    for name in ("unit_moments.npz", "field_fluctuations.npz", "population_variance.csv"):
        path = _source_bound_file(root, manifest, "analysis/"+name)
        require(report["outputs"][name]["sha256"] == sha256_file(path)
                and report["outputs"][name]["bytes"] == path.stat().st_size,
                "reported fluctuation output changed")
        files[name] = path
    with np.load(files["unit_moments.npz"], allow_pickle=False) as stored:
        units = {key: stored[key] for key in stored.files}
    with np.load(files["field_fluctuations.npz"], allow_pickle=False) as stored:
        fields = {key: stored[key] for key in stored.files}
    mapping, sizes = unit_mapping(n_e, n_i, 16)
    require(set(units) == {"unit_ids", "unit_field_indices"} |
            {condition+"_"+moment+"_counts" for condition in CONDITIONS for moment in ("mean", "variance")}
            and units["unit_ids"].dtype.kind in "iu"
            and units["unit_field_indices"].dtype.kind in "iu"
            and np.array_equal(units["unit_ids"], np.arange(total))
            and np.array_equal(units["unit_field_indices"], mapping),
            "complete unit identities or moment inventory changed")
    require(set(fields) == {"field_unit_counts", "source_rng_seeds"} |
            {condition+"_"+suffix for condition in CONDITIONS
             for suffix in ("field_counts", "field_covariance_counts")}
            and fields["field_unit_counts"].dtype.kind in "iu"
            and fields["source_rng_seeds"].dtype.kind in "iu"
            and np.array_equal(fields["field_unit_counts"], sizes)
            and np.array_equal(fields["source_rng_seeds"], seeds)
            and report["field_unit_counts"] == sizes.tolist(), "field or block map changed")
    for condition in CONDITIONS:
        for moment in ("mean", "variance"):
            x = units[condition+"_"+moment+"_counts"]
            require(x.shape == (len(PHASES), total) and x.dtype.kind == "f"
                    and np.isfinite(x).all() and (x >= 0).all(), "invalid all-unit moments")
        x, cov = fields[condition+"_field_counts"], fields[condition+"_field_covariance_counts"]
        require(x.shape == (len(seeds), 1000, 17) and x.dtype.kind in "iu" and (x >= 0).all()
                and cov.shape == (len(PHASES), 17, 17) and np.isfinite(cov).all(),
                "incomplete field trajectory or covariance")
    with files["population_variance.csv"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    wanted = {(condition, phase, population) for condition in CONDITIONS
              for phase in PHASES for population in ("E", "I", "all")}
    indexed = {}
    for row in rows:
        key = row["condition"], row["phase"], row["population"]
        require(key in wanted and key not in indexed and int(row["joint_rng_blocks"]) == len(seeds)
                and float(row["bin_ms"]) == 5.
                and int(row["units"]) == {"E": n_e, "I": n_i, "all": total}[key[2]]
                and int(row["time_bins"]) == round((PHASES[key[1]][1]-PHASES[key[1]][0])/5),
                "population row identity or denominator changed")
        indexed[key] = {metric: None if row[metric] == "" else float(row[metric]) for metric in METRICS}
        require(all(value is None or np.isfinite(value) for value in indexed[key].values()),
                "nonfinite population statistic")
    require(set(indexed) == wanted, "complete population comparison rows required")
    cohort = dict(root=root, native=native, config=config, report=report, units=units, fields=fields,
                sizes=sizes, rows=indexed, delivery_sha256=sha256_file(root/"delivery.json"),
                count_histograms=load_count_histograms(root,manifest,report,config,fields,sizes,seeds))
    cohort['phase_responses'] = None if native else pfc_phase_responses.load(cohort,manifest)
    return cohort


def load_count_histograms(root, manifest, report, config, fields, sizes, seeds):
    name = 'unit_count_histograms.npz'
    if name not in report['outputs']:
        return None
    path = _source_bound_file(root, manifest, 'analysis/'+name)
    binding = report['outputs'][name]
    require(binding['sha256'] == sha256_file(path) and binding['bytes'] == path.stat().st_size,
            'reported count histogram changed')
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    require(set(data) == {'coordinates','observations','coordinate_axes','condition_names',
        'source_rng_seeds','phase_names','phase_bounds_ms','window_ms','field_unit_counts','expected_observations'}
        and np.array_equal(data['coordinate_axes'], pfc_count_histograms.AXES)
        and np.array_equal(data['condition_names'], CONDITIONS)
        and np.array_equal(data['phase_names'], list(PHASES))
        and np.array_equal(data['phase_bounds_ms'], list(PHASES.values()))
        and data['source_rng_seeds'].dtype.kind in 'iu'
        and np.array_equal(data['source_rng_seeds'], seeds)
        and data['field_unit_counts'].dtype.kind in 'iu'
        and np.array_equal(data['field_unit_counts'], sizes), 'count histogram axes or source identities changed')
    windows = data['window_ms']
    require(windows.ndim == 1 and len(windows) > 0 and windows.dtype.kind in 'ifu'
            and np.isfinite(windows).all() and np.all(windows > 0)
            and np.all(windows[1:] > windows[:-1])
            and np.all(windows/config['readout_bin_ms'] == np.rint(windows/config['readout_bin_ms'])),
            'complete registered count window grid required')
    shape = (len(CONDITIONS),len(seeds),len(PHASES),len(windows),len(sizes))
    index, mass = data['coordinates'], data['observations']
    limit = np.iinfo(np.int64).max
    require(index.ndim == 2 and index.shape[1] == 6 and len(index) > 0 and index.dtype.kind in 'iu'
            and mass.shape == (len(index),) and mass.dtype.kind in 'iu'
            and np.all(index >= 0) and np.all(mass > 0)
            and int(index.max()) <= limit and int(mass.max()) <= limit
            and np.all(index[:,:5] < np.asarray(shape))
            and len(np.unique(index,axis=0)) == len(index), 'unique nonnegative sparse count identities required')
    index, mass = index.astype(np.int64), mass.astype(np.int64)
    require(np.all(index[:,5] <= limit//mass), 'count histogram weighted sums exceed int64')
    observed = np.zeros(shape,dtype=np.int64)
    spikes = np.zeros_like(observed)
    np.add.at(observed,tuple(index[:,:5].T),mass)
    np.add.at(spikes,tuple(index[:,:5].T),mass*index[:,5])
    expected = np.zeros_like(observed)
    for ci, condition in enumerate(CONDITIONS):
        for pi, (start,stop) in enumerate(PHASES.values()):
            truth = fields[condition+'_field_counts'][:,round(start/5):round(stop/5)].sum(axis=1)
            for wi,width in enumerate(windows):
                number = round((stop-start)/width)
                require(number > 0 and number*width == stop-start, 'whole phase count windows required')
                expected[ci,:,pi,wi] = number*sizes
                require(np.array_equal(spikes[ci,:,pi,wi],truth),
                        'count histogram spike totals disagree with independently verified field counts')
    require(data['expected_observations'].dtype.kind in 'iu'
            and np.array_equal(data['expected_observations'],expected)
            and np.array_equal(observed,expected), 'count histogram unit/window mass changed')
    summary = report['unit_count_histograms']
    require(summary['blocks'] == len(seeds) and summary['field_count'] == len(sizes)
            and summary['sparse_rows'] == len(index)
            and summary['unit_window_observations'] == int(observed.sum())
            and summary['window_ms'] == windows.tolist()
            and summary['observations_are_independent_replicates'] is False,
            'count histogram summary or replication meaning changed')
    return dict(coordinates=index,observations=mass,shape=shape,windows_ms=windows,
                sha256=binding['sha256'],mass_checks=int(observed.size),spike_checks=int(spikes.size))


def compare_count_cdfs(clock_cohorts, windows_ms):
    require(isinstance(windows_ms,list) and windows_ms
            and all(type(x) in (int,float) and np.isfinite(x) and x > 0 for x in windows_ms)
            and windows_ms == sorted(set(windows_ms)), 'declared count CDF window grid required')
    for cohort in clock_cohorts.values():
        require(cohort['count_histograms'] is not None
                and np.array_equal(cohort['count_histograms']['windows_ms'],windows_ms),
                'all source clocks need the complete declared count histograms')
    support = np.unique(np.concatenate([cohort['count_histograms']['coordinates'][:,5]
                                        for cohort in clock_cohorts.values()]))
    cdfs = {}
    for label,cohort in clock_cohorts.items():
        value = cohort['count_histograms']
        cdfs[label] = pfc_count_histograms.cdf(value['coordinates'],value['observations'],support,value['shape'])
    labels = list(clock_cohorts)
    rows, distances = [], []
    for left,right in zip(labels[:-1],labels[1:]):
        a,b = cdfs[left].mean(axis=1),cdfs[right].mean(axis=1)
        require(a.shape == b.shape, 'matching complete count CDF axes required')
        blocks = len(clock_cohorts[left]['report']['joint_rng_blocks'])
        for ci,condition in enumerate(CONDITIONS):
            for pi,phase in enumerate(PHASES):
                for wi,width in enumerate(windows_ms):
                    for fi,size in enumerate(clock_cohorts[left]['sizes']):
                        first,second = a[ci,pi,wi,fi],b[ci,pi,wi,fi]
                        context = dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                            window_ms=width,field_index=fi,field_units=int(size),joint_rng_blocks=blocks)
                        for si,threshold in enumerate(support):
                            rows.append(dict(context,count_threshold=int(threshold),
                                left_mean_cdf=float(first[si]),right_mean_cdf=float(second[si]),
                                left_minus_right=float(first[si]-second[si])))
                        distances.append(dict(context,
                            max_abs_empirical_cdf_difference=float(np.max(np.abs(first-second))),
                            status='DESCRIPTIVE_NOT_EQUIVALENCE'))
    common = next(iter(clock_cohorts.values()))
    payload = {label+'_cdf':array for label,array in cdfs.items()}
    payload.update(count_support=support,condition_names=np.asarray(CONDITIONS),
        source_rng_seeds=np.asarray(common['report']['joint_rng_blocks'],dtype=np.int64),
        phase_names=np.asarray(list(PHASES)),window_ms=np.asarray(windows_ms),
        field_unit_counts=common['sizes'])
    return rows,distances,payload,dict(
        status='COMPLETE_DESCRIPTIVE_COUNT_CDF_COMPARISON',compared_clocks=len(clock_cohorts),
        count_support=support.tolist(),independent_joint_blocks=payload['source_rng_seeds'].tolist(),
        unit_window_mass_checks=sum(c['count_histograms']['mass_checks'] for c in clock_cohorts.values()),
        independent_field_spike_total_checks=sum(c['count_histograms']['spike_checks'] for c in clock_cohorts.values()),
        interpretation='Equal-weight whole-block mean CDFs of marginal unit/window counts within each source field. Individual block CDFs are retained. Conservation checks do not establish timing, dependence or full joint-distribution fidelity. No confidence interval or equivalence gate.')

def compare(clock_archives, native_reference, config_path, output, precision_config_path=None):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", new_source_simulations=0, production_episodes=0,
                  scientific_fidelity_qualified=False, distributional_convergence_qualified=False,
                  finite_size_scaling_qualified=False, confidence_intervals=False, p_values=False, q_values=False)
    try:
        settings = yaml.safe_load(Path(config_path).read_text())
        lags = settings["lag_ms"]
        require(settings["scope"] == "development_calibration" and settings["bin_ms"] == 5.
                and settings["lag_scope"] == "complete_within_phase_pairs"
                and settings["inference"] == "descriptive_only"
                and isinstance(lags, list) and lags and len(lags) == len(set(lags)) and 0 in lags
                and all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                        and value/5 == round(value/5) for value in lags),
                "predeclared complete-bin lag grid required")
        require(len(clock_archives) == 3, "exactly three clock archives required")
        loaded = [load_cohort(path) for path in clock_archives]
        clocks = {cohort["config"]["dt_ms"]: cohort for cohort in loaded}
        require(set(clocks) == set(CLOCKS) and len(clocks) == 3, "distinct registered clocks required")
        reference_config = {key: value for key, value in clocks[.05]["config"].items() if key != "dt_ms"}
        for cohort in clocks.values():
            require({key: value for key, value in cohort["config"].items() if key != "dt_ms"}
                    == reference_config, "only clock may differ between large cohorts")
            a = clocks[.05]["report"]["common_execution_binding"]
            b = cohort["report"]["common_execution_binding"]
            require(a["backend"] == b["backend"] == "numpy"
                    and all(a["code"][key] == b["code"][key] for key in ("executor_sha256", "drive_sha256")),
                    "large source backend or dynamics changed")
        precision_settings = None
        if precision_config_path is not None:
            require('count_windows_ms' in settings, 'count CDF observations required for calibration precision')
            precision_settings = yaml.safe_load(Path(precision_config_path).read_text())
            pfc_numerical_precision.validate_settings(precision_settings,clocks)
        native = load_cohort(native_reference, native=True)
        require(all(native["config"][key] == clocks[.05]["config"][key] for key in PHYSICAL_KEYS),
                "shared native/large physical settings changed")
        labels = {clock: "numpy_dt_"+str(clock).replace(".", "p") for clock in CLOCKS}
        cohorts = {labels[clock]: clocks[clock] for clock in CLOCKS}
        cohorts["native_2560"] = native
        payload, covariances, cohort_rows = {}, {}, []
        for label, cohort in cohorts.items():
            scale = cohort["sizes"]*.005
            for condition in CONDITIONS:
                rates = cohort["fields"][condition+"_field_counts"] / scale
                for index, (phase, (start, stop)) in enumerate(PHASES.items()):
                    selected = rates[:, round(start/5):round(stop/5)]
                    for lag_ms in lags:
                        covariance = lag_covariance(selected, round(lag_ms/5))
                        covariances[label, condition, phase, lag_ms] = covariance
                        if covariance is not None:
                            payload[f"{label}_{condition}_{phase}_lag_{lag_ms:g}ms"] = covariance
                        if lag_ms == 0:
                            original = cohort["fields"][condition+"_field_covariance_counts"][index] / np.outer(scale, scale)
                            require(np.allclose(covariance, original, rtol=1e-10, atol=1e-12),
                                    "normalized zero-lag covariance differs from upstream result")
            for (condition, phase, population), values in cohort["rows"].items():
                cohort_rows.append(dict(cohort=label, condition=condition, phase=phase,
                    population=population, coupled_units=cohort["report"]["coupled_core_units"],
                    dt_ms=cohort["config"]["dt_ms"], **values))
        comparisons = [(labels[.05], labels[.025], "adjacent_clock"),
                       (labels[.025], labels[.0125], "adjacent_clock"),
                       ("native_2560", labels[.05], "matched_clock_size_description")]
        population_rows, moment_rows, covariance_rows = [], [], []
        for left, right, kind in comparisons:
            a, b = cohorts[left], cohorts[right]
            for key in a["rows"]:
                condition, phase, population = key
                for metric in METRICS:
                    x, y = a["rows"][key][metric], b["rows"][key][metric]
                    population_rows.append(dict(comparison=kind, left_cohort=left, right_cohort=right,
                        condition=condition, phase=phase, population=population, metric=metric,
                        left_value=x, right_value=y, left_minus_right=None if x is None or y is None else x-y,
                        status="UNDEFINED" if x is None or y is None else "DEFINED"))
            for condition in CONDITIONS:
                for index, phase in enumerate(PHASES):
                    if kind == "adjacent_clock":
                        for moment in ("mean", "variance"):
                            x, y = (cohort["units"][condition+"_"+moment+"_counts"][index] for cohort in (a, b))
                            moment_rows.append(dict(left_cohort=left, right_cohort=right, condition=condition,
                                phase=phase, moment=moment, compared_units=len(x), **difference_summary(x, y)))
                    for lag_ms in lags:
                        covariance_rows.append(dict(comparison=kind, left_cohort=left, right_cohort=right,
                            condition=condition, phase=phase, lag_ms=lag_ms, field_pairs=17*17,
                            covariance_units="Hz^2",
                            **difference_summary(covariances[left, condition, phase, lag_ms],
                                                 covariances[right, condition, phase, lag_ms])))
        np.savez_compressed(output/"lagged_field_covariance_hz2.npz", **payload)
        tables = {"by_cohort_population_variance.csv": cohort_rows,
                  "population_differences.csv": population_rows,
                  "unit_moment_clock_differences.csv": moment_rows,
                  "lagged_field_covariance_differences.csv": covariance_rows}
        extra_outputs = []
        cdf_payload = None
        if 'count_windows_ms' in settings:
            selected = {labels[clock]:clocks[clock] for clock in CLOCKS}
            cdf_rows,cdf_distances,cdf_payload,cdf_report = compare_count_cdfs(selected,settings['count_windows_ms'])
            tables['count_cdf_differences.csv'] = cdf_rows
            tables['count_cdf_descriptive_distances.csv'] = cdf_distances
            np.savez_compressed(output/'block_count_cdfs.npz',**cdf_payload)
            extra_outputs.append('block_count_cdfs.npz')
            result['count_cdf_comparison'] = cdf_report
        selected = {labels[clock]:clocks[clock] for clock in CLOCKS}
        if any(c['phase_responses'] is not None for c in selected.values()):
            response_tables,response_payload,response_report = pfc_phase_responses.compare(selected)
            tables.update(response_tables)
            np.savez_compressed(output/'block_source_responses.npz',**response_payload)
            extra_outputs.append('block_source_responses.npz')
            result['source_phase_responses'] = response_report
        if precision_settings is not None:
            require(all(c['phase_responses'] is not None for c in selected.values()),
                    'complete source phase responses required for calibration precision')
            precision_tables,precision_payload,precision_report = pfc_numerical_precision.analyze(
                selected,cdf_payload,covariances,lags,precision_settings)
            tables.update(precision_tables)
            np.savez_compressed(output/'calibration_bootstrap.npz',**precision_payload)
            extra_outputs.append('calibration_bootstrap.npz')
            precision_report['settings_sha256'] = sha256_file(precision_config_path)
            precision_report['implementation_sha256'] = sha256_file(Path(pfc_numerical_precision.__file__))
            result['calibration_precision'] = precision_report
        for name, rows in tables.items():write_rows(output/name, rows)
        case_counts = {label:cohort['report']['completed_source_cases'] for label,cohort in cohorts.items()}
        block_counts = {label:len(cohort['report']['joint_rng_blocks']) for label,cohort in cohorts.items()}
        result.update(status="COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON",
            settings=settings, settings_sha256=sha256_file(config_path),
            source_episodes_reused=sum(case_counts.values()), registered_cases_by_cohort=case_counts,
            registered_cases_per_cohort=next(iter(case_counts.values())) if len(set(case_counts.values())) == 1 else None,
            cohorts=len(cohorts), independent_joint_blocks_by_cohort=block_counts,
            independent_joint_blocks_per_condition=next(iter(block_counts.values())) if len(set(block_counts.values())) == 1 else None, unitwise_comparisons="adjacent equal-size clocks only",
            covariance_orientation="row: leading field at t; column: trailing field at t+lag",
            covariance_estimator="across-block matched-time sample covariance, averaged over complete within-phase pairs",
            covariance_units="Hz^2", undefined_phase_lag_rows=sum(row["status"] != "DEFINED" for row in covariance_rows),
            inputs={label: dict(archive=str(cohort["root"]), delivery_sha256=cohort["delivery_sha256"],
                coupled_units=cohort["report"]["coupled_core_units"], dt_ms=cohort["config"]["dt_ms"],
                source_rng_blocks=cohort["report"]["joint_rng_blocks"],
                weight_scaling_factor=cohort["config"]["weight_scaling_factor"],
                field_unit_counts=cohort["sizes"].tolist(),
                backend=cohort["report"].get("source_backend", "numpy"),
                source_versions=cohort["report"].get("source_versions",
                    cohort["report"].get("common_execution_binding", {}).get("driver", {})))
                for label, cohort in cohorts.items()},
            interpretation="Clock differences combine discretization and different native stochastic paths. The matched-clock size description also changes backend and RNG cohort. Development blocks, time bins and units do not establish equivalence, a finite-size law or independent confirmation.",
            row_counts={name: len(rows) for name, rows in tables.items()},
            implementation_sha256=sha256_file(Path(__file__)),
            outputs={name: dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name))
                     for name in (*tables, "lagged_field_covariance_hz2.npz", *extra_outputs)})
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        write_json(output/"report.json", result)
        raise
    write_json(output/"report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-archive", action="append", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path)
    args = parser.parse_args(argv)
    compare(args.clock_archive, args.native_reference, args.config, args.output, args.precision_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

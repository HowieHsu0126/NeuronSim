"""Describe full-unit integrated-count covariance from retained ensembles.

Adjacent disjoint draw pairs are fixed before reading outcomes. Paired
differences remove unknown means. Gram matrices retain every off-diagonal unit
pair without allocating a unit-by-unit matrix. Signed U-statistics correct
finite-ensemble noise; delete-one-pair ranges are sensitivity, not confidence
intervals. Native and learned conditioning differ in the saved development case.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .provenance import sha256_file

PERMUTATION_SEED = 73103


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def paired_differences(totals: np.ndarray) -> np.ndarray:
    values = np.asarray(totals, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] >= 6 and values.shape[0] % 2 == 0
             and values.shape[1] >= 2 and np.isfinite(values).all(),
             "finite even draw-by-unit array with at least six draws required")
    return (values[::2] - values[1::2]) / np.sqrt(2.)


def off_diagonal_kernel(left: np.ndarray, right: np.ndarray, *, chunk_units: int = 4096) -> np.ndarray:
    """Mean off-diagonal inner product of paired-difference outer products.

    For independent vectors a,b, E[k(a,b)] = mean_{u != v} SigmaA_uv
    SigmaB_uv. Products from the same vector are used only in empirical energy.
    """
    _require(left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1]
             and left.shape[1] >= 2 and np.isfinite(left).all() and np.isfinite(right).all(),
             "finite paired-difference matrices on identical units required")
    _require(type(chunk_units) is int and chunk_units > 0, "positive integer unit chunk required")
    gram = np.zeros((len(left), len(right)), dtype=np.float64)
    diagonal = np.zeros_like(gram)
    for start in range(0, left.shape[1], chunk_units):
        a = left[:, start:start + chunk_units]
        b = right[:, start:start + chunk_units]
        gram += a @ b.T
        diagonal += (a * a) @ (b * b).T
    result = (gram * gram - diagonal) / (left.shape[1] * (left.shape[1] - 1))
    _require(np.isfinite(result).all(), "nonfinite covariance kernel")
    return result


def covariance_summary(aa: np.ndarray, bb: np.ndarray, ab: np.ndarray) -> dict[str, float]:
    """Separate empirical covariance difference from sampling noise.

    U = mean distinct-vector kernel; V = mean all-vector kernel. V-U estimates
    covariance-estimation variance. Cross-law ensembles must be independent.
    """
    _require(aa.ndim == bb.ndim == ab.ndim == 2 and aa.shape == (len(aa), len(aa))
             and bb.shape == (len(bb), len(bb)) and ab.shape == (len(aa), len(bb))
             and min(len(aa), len(bb)) >= 2 and all(np.isfinite(x).all() for x in (aa, bb, ab)),
             "matching finite within-law and cross-law kernel matrices required")
    def energy(kernel):
        n = len(kernel)
        empirical = float(kernel.mean())
        corrected = float((kernel.sum() - np.trace(kernel)) / (n * (n - 1)))
        return empirical, corrected, empirical - corrected
    va, ua, na = energy(aa)
    vb, ub, nb = energy(bb)
    cross = float(ab.mean())
    return dict(native_empirical_energy=va, learned_empirical_energy=vb,
                native_corrected_energy=ua, learned_corrected_energy=ub,
                native_sampling_noise=na, learned_sampling_noise=nb,
                cross_covariance_inner_product=cross,
                empirical_difference=va + vb - 2 * cross,
                sampling_noise=na + nb,
                corrected_difference=ua + ub - 2 * cross)


def describe_totals(native: np.ndarray, learned: np.ndarray, *, chunk_units: int = 4096):
    _require(native.shape == learned.shape, "matched draw/unit coverage required")
    a, b = paired_differences(native), paired_differences(learned)
    aa = off_diagonal_kernel(a, a, chunk_units=chunk_units)
    bb = off_diagonal_kernel(b, b, chunk_units=chunk_units)
    ab = off_diagonal_kernel(a, b, chunk_units=chunk_units)
    summary = covariance_summary(aa, bb, ab)
    sensitivity = []
    for law in ("native", "learned"):
        for deleted in range(len(a)):
            keep = np.arange(len(a)) != deleted
            value = (covariance_summary(aa[np.ix_(keep, keep)], bb, ab[keep]) if law == "native"
                     else covariance_summary(aa, bb[np.ix_(keep, keep)], ab[:, keep]))
            sensitivity.append(dict(deleted_law=law, deleted_pair=deleted, **value))
    values = [row["corrected_difference"] for row in sensitivity]
    summary.update(delete_one_pair_minimum=min(values), delete_one_pair_maximum=max(values))
    return summary, sensitivity, {"native": aa, "learned": bb, "cross": ab}


def empirical_covariance_energy(totals: np.ndarray) -> float:
    """Mean squared off-diagonal sample covariance, with ddof=1."""
    values = np.asarray(totals, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 2
             and np.isfinite(values).all(), "finite draw-by-unit counts required")
    centered = values - values.mean(axis=0)
    kernel = off_diagonal_kernel(centered, centered)
    return float(kernel.sum() / (len(values) - 1) ** 2)


def shuffle_unit_draws(totals: np.ndarray) -> np.ndarray:
    """Fixed independent permutations preserve every unit's sampled marginal.

    Resetting this generator for each horizon preserves the same unit/draw
    permutation across nested horizons. These are reused outputs, not new draws
    from the mechanistic or learned model. No independence from the source
    ensemble is asserted.
    """
    _require(totals.ndim == 2, "draw-by-unit totals required")
    return np.random.Generator(np.random.PCG64(PERMUTATION_SEED)).permuted(totals, axis=0)


def describe_law(totals: np.ndarray):
    differences = paired_differences(totals)
    kernel = off_diagonal_kernel(differences, differences)
    n = len(differences)
    empirical = float(kernel.mean())
    corrected = float((kernel.sum() - np.trace(kernel)) / (n * (n - 1)))
    sensitivity = []
    for deleted in range(n):
        keep = np.arange(n) != deleted
        sub = kernel[np.ix_(keep, keep)]
        sensitivity.append(dict(deleted_pair=deleted,
            corrected_energy=float((sub.sum() - np.trace(sub)) / ((n - 1) * (n - 2)))))
    shuffled = shuffle_unit_draws(totals)
    # The two empirical energies share observed marginals. No independent-law
    # covariance-distance correction is valid for this paired permutation.
    original_energy = empirical_covariance_energy(totals)
    shuffled_energy = empirical_covariance_energy(shuffled)
    row = dict(paired_outer_product_empirical_energy=empirical, corrected_energy=corrected,
               sampling_noise=empirical - corrected,
               sample_covariance_empirical_energy=original_energy,
               shuffled_sample_covariance_empirical_energy=shuffled_energy,
               empirical_energy_minus_shuffled=original_energy - shuffled_energy,
               delete_pair_minimum=min(r['corrected_energy'] for r in sensitivity),
               delete_pair_maximum=max(r['corrected_energy'] for r in sensitivity))
    return row, sensitivity, kernel


def reuse_retained_analysis(root: Path):
    """Read delivered per-cell descriptions; the caller authenticates the receipt."""
    report = json.loads((root / 'analysis.json').read_text())
    _require(report['status'] == 'DESCRIBED_RETAINED_POPULATION_COVARIANCE_DEVELOPMENT'
             and report['scientific_claim_qualified'] is False and report['formal_test_accessed'] is False
             and report['case_id'] == 'train_block_014_familiar_0' and report['bin_ms'] == 5.
             and report['horizons_ms'] == [20, 80, 500] and report['permutation_seed'] == PERMUTATION_SEED
             and report['draw_pairs'] == [[i, i + 1] for i in range(0, 16, 2)],
             'matching completed retained analysis protocol required')
    for name, digest in report['artifacts'].items():
        _require(sha256_file(root / name) == digest, 'completed retained analysis changed: ' + name)
    with (root / 'retained_unit_covariance.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    # CSV values retain types by matching the authenticated JSON summary.
    _require(len(rows) == len(report['summary']) and all(
        all(str(value) == row[name] for name, value in summary.items())
        for row, summary in zip(rows, report['summary'])), 'completed covariance table and summary differ')
    with (root / 'delete_pair_sensitivity.csv').open() as stream:
        deleted = list(csv.DictReader(stream))
    with np.load(root / 'covariance_kernels.npz', allow_pickle=False) as array:
        kernels = {k: array[k] for k in array.files}
    _require(len(report['cells']) * 3 == len(rows) and len(deleted) == len(rows) * 8,
             'complete retained cell/horizon sensitivity coverage required')
    return report, deleted, kernels


def analyze_retained_cells(cells: list[Path], output: Path, *, previous_analyses: list[Path] | None = None) -> dict:
    """Describe authenticated retained learned counts, without source-law comparison.

    The caller authenticates deliveries and stages only the first retained
    case. Complete three-seed coverage is required by the campaign runner;
    one completed cell is also usable for implementation preparation.
    """
    rows, deleted, kernels, bindings = [], [], {}, []
    source_identity = None
    source_units = None
    reused = []
    for previous in previous_analyses or []:
        report, sensitivity, gram = reuse_retained_analysis(previous)
        source_identity = report['source_identity'] if source_identity is None else source_identity
        _require(report['source_identity'] == source_identity, 'reused retained cohort identity differs')
        old_keys = {(r['representation'], r['seed']) for r in bindings}
        new_keys = {(r['representation'], r['seed']) for r in report['cells']}
        _require(not old_keys & new_keys and len(new_keys) == len(report['cells']), 'duplicate reused retained cells')
        rows.extend(report['summary']); deleted.extend(sensitivity); kernels.update(gram); bindings.extend(report['cells'])
        reused.append(dict(analysis_sha256=sha256_file(previous / 'analysis.json'), input=str(previous), cells=list(sorted(new_keys))))
    for cell in cells:
        evaluation = json.loads((cell / 'forecasts/evaluation.json').read_text())
        run = json.loads((cell / 'training/run.json').read_text())
        replay = json.loads((cell / 'replay.json').read_text())
        seed, representation = run['training_seed'], run['representation']
        key = (representation, seed)
        _require(key not in [(r['representation'], r['seed']) for r in bindings], "duplicate retained representation/seed")
        cfg = evaluation['configuration']
        _require(run['status'] == 'FITTED_DEVELOPMENT' and run['model_family'] == 'population'
                 and seed in (41, 42, 43) and representation in ('fixed', 'recurrent')
                 and evaluation['status'] == 'EVALUATED_DEVELOPMENT' and evaluation['model_type'] == 'population'
                 and evaluation['training_seed'] == replay['training_seed'] == seed
                 and cfg['joint_samples'] == 16 and cfg['horizons_ms'] == [20., 80., 500.]
                 and cfg['origins_ms'] == [1000., 3000., 5000.] and cfg['formal_evaluation'] is False
                 and replay['status'] == 'PASS' and replay['model_type'] == 'population'
                 and replay['verified_origins'] == 48 and replay['verified_score_rows'] == 144
                 and replay['all_forecast_unit_metrics_replayed'] is True,
                 "completed source-bound population evaluation and independent replay required")
        first = replay['first_origin_exact_replay']
        sample_path = cell / 'forecasts/full_unit_samples.npz'
        sample_hash = sha256_file(sample_path)
        _require(first['status'] == 'EXACT_FULL_UNIT_AND_INDEPENDENT_FIELD_REPLAY'
                 and first['case_id'] == 'train_block_014_familiar_0' and first['origin_ms'] == 1000.
                 and first['unit_count_values'] == 160000000 and first['unit_ids'] == 100000
                 and first['full_unit_artifact_sha256'] == evaluation['artifacts']['full_unit_samples.npz'] == sample_hash
                 and replay['evaluation_sha256'] == sha256_file(cell / 'forecasts/evaluation.json')
                 and run['best_checkpoint_sha256'] == evaluation['checkpoint_sha256'] == replay['source_checkpoint_sha256']
                 and evaluation['native_source_identity'] == run['binding']['native_source_identity']
                 and evaluation['scientific_claim_qualified'] is False and evaluation['formal_test_accessed'] is False,
                 "retained full-unit counts/checkpoint/source binding changed")
        identity = dict(native_source_identity=evaluation['native_source_identity'], dataset_sha256=evaluation['dataset_sha256'],
                        acquisition_design_sha256=evaluation['acquisition_design_sha256'])
        if source_identity is None:
            source_identity = identity
        _require(identity == source_identity, "retained cells use different native cohort identities")
        with np.load(sample_path, allow_pickle=False) as samples:
            units = samples['unit_ids']
            _require(units.shape == (100000,) and units.dtype.kind in 'iu' and len(np.unique(units)) == 100000,
                     "complete unique100000 source unit identities required")
            if source_units is None:
                source_units = units
            _require(np.array_equal(units, source_units), 'retained cells use different source unit order')
            counts = samples['sampled_unit_counts']
            _require(counts.shape == (16, 100, 100000), "complete retained16draw count array required")
            totals = horizon_totals(counts, (4, 16, 100))
        for bins in (4, 16, 100):
            result, sensitivity, kernel = describe_law(totals[bins])
            context = dict(representation=representation, seed=seed, case_id=first['case_id'],
                           origin_ms=1000., horizon_ms=5 * bins, unit_count=100000, draws=16, draw_pairs=8)
            rows.append({**context, **result})
            deleted.extend({**context, **r} for r in sensitivity)
            kernels[f'{representation}_{seed}_bins{bins}'] = kernel
        bindings.append(dict(representation=representation, seed=seed, input=str(cell),
                             sample_sha256=sample_hash, checkpoint_sha256=evaluation['checkpoint_sha256'],
                             evaluation_sha256=sha256_file(cell / 'forecasts/evaluation.json'),
                             replay_sha256=sha256_file(cell / 'replay.json')))
    _require(bool(rows), "at least one completed retained cell required")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / 'retained_unit_covariance.csv', rows)
    _write_csv(output / 'delete_pair_sensitivity.csv', deleted)
    np.savez(output / 'covariance_kernels.npz', **kernels)
    report = dict(status='DESCRIBED_RETAINED_POPULATION_COVARIANCE_DEVELOPMENT', cells=bindings,
        reused_analyses=reused,
        source_identity=source_identity, case_id='train_block_014_familiar_0', native_initial_noise_blocks=1,
        connectivity_realizations=1, bin_ms=5., horizons_ms=[20, 80, 500], units='spike_count^4',
        draw_pairs=[[i, i + 1] for i in range(0, 16, 2)], permutation_seed=PERMUTATION_SEED,
        permutation_algorithm='numpy.Generator(PCG64).permuted(axis=0); same permutation for each horizon',
        sample_covariance_ddof=1, shuffle_scope='each unit total retains its entire empirical marginal; cross-unit draw alignment permuted',
        shuffle_interpretation='empirical paired energy difference only; shuffled counts reuse samples and are not independent draws',
        covariance_scope='integrated counts; temporal covariance not evaluated',
        conditioning='learned observed-history posterior; no native conditional truth used',
        uncertainty='delete-one-disjoint-pair sensitivity only; no CI, p-value, FDR or significance',
        fidelity_interpretation='describes generated dependence and sampling noise; no fidelity or closure qualification',
        training_performed=False, generation_performed=False, simulation_performed=False,
        formal_test_accessed=False, scientific_claim_qualified=False, summary=rows,
        artifacts={name: sha256_file(output / name) for name in
                   ('retained_unit_covariance.csv', 'delete_pair_sensitivity.csv', 'covariance_kernels.npz')},
        implementation_sha256=sha256_file(Path(__file__)))
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def horizon_totals(counts: np.ndarray, horizons_bins: tuple[int, ...], *, chunk_units: int = 4096):
    _require(counts.ndim == 3 and counts.dtype.kind in "iu" and counts.shape[2] >= 2,
             "integer draw-by-bin-by-unit counts required")
    _require(type(chunk_units) is int and chunk_units > 0 and horizons_bins
             and all(type(h) is int and 0 < h <= counts.shape[1] for h in horizons_bins),
             "valid cumulative bin horizons and positive unit chunk required")
    totals = {h: np.empty((len(counts), counts.shape[2]), dtype=np.float64) for h in horizons_bins}
    # Validate the full saved array, including bins outside a requested horizon.
    # Float64 sums avoid integer overflow. Counts beyond exact float64 support
    # are invalid for this numerical path rather than silently rounded.
    bound = (2 ** 53 - 1) // counts.shape[1]
    for start in range(0, counts.shape[2], chunk_units):
        block = counts[:, :, start:start + chunk_units]
        _require(np.all(block >= 0) and np.all(block <= bound), "nonnegative exactly summable counts required")
        for h in horizons_bins:
            totals[h][:, start:start + chunk_units] = block[:, :h].sum(axis=1, dtype=np.float64)
    return totals


def validate_alignment(native_report: dict, learned_report: dict, native_fields: dict, learned_fields: dict) -> None:
    _require(native_report["status"] == "DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT"
             and learned_report["status"] == "EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT"
             and native_report["draws"] == learned_report["draws"] == 16
             and native_report["boundary_tick"] == learned_report["boundary_tick"] == 10000
             and native_report["generated_bins"] == learned_report["generated_bins"] == 100
             and native_report["reference_unit_count"] == learned_report["unit_count"] == 100000
             and native_report["field_count"] == learned_report["field_count"] == 128
             and learned_report["training_seed"] == 41,
             "completed shared100000-unit seed41 boundary/horizon coverage required")
    config = learned_report["configuration"]
    _require(config["case_id"] == "train_block_014_familiar_0" and config["bin_ms"] == 5.
             and config["native_dt_ms"] == .1 and config["boundary_time_ms"] == 1000.
             and config["horizon_ms"] == 500. and config["formal_evaluation"] is False,
             "identified development case, bins and interval required")
    native_binding, learned_binding = native_report["reference_binding"], learned_report["source_binding"]
    _require(all(native_binding.get(k) == v for k, v in learned_binding.items())
             and native_binding["fitting_scale_sha256"] == learned_report["fitting_scale_sha256"]
             and native_report["reference_source_identity"] == learned_report["native_source_identity"],
             "shared native episode/source/fitting-scale binding required")
    for name in ("unit_ids", "unit_field_index", "field_ids", "unit_count", "observed_unit_counts", "observed_field_counts"):
        _require(np.array_equal(native_fields[name], learned_fields[name]), "saved observation identity differs: " + name)
    units, mapping = native_fields["unit_ids"], native_fields["unit_field_index"]
    _require(units.shape == mapping.shape == (100000,) and units.dtype.kind in "iu"
             and len(np.unique(units)) == 100000 and mapping.dtype.kind in "iu"
             and np.all((mapping >= 0) & (mapping < 128))
             and native_fields["field_ids"].shape == (128,)
             and len(np.unique(native_fields["field_ids"])) == 128
             and np.array_equal(np.bincount(mapping, minlength=128), native_fields["unit_count"])
             and native_fields["observed_unit_counts"].shape == (100, 100000),
             "complete unique source units and field denominator required")
    # The learned producer stores a singleton batch axis for the same control.
    _require(learned_fields["stimulus"].shape == (1, *native_fields["stimulus"].shape)
             and np.array_equal(native_fields["stimulus"], learned_fields["stimulus"][0]),
             "saved future stimulus differs")
    _require(native_report["scientific_claim_qualified"] is False
             and learned_report["scientific_claim_qualified"] is False,
             "development receipts required")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(native: Path, learned: Path, output: Path) -> dict:
    """Consume staged authenticated producer outputs without new generation."""
    started = time.perf_counter()
    nr = json.loads((native / "analysis.json").read_text())
    lr = json.loads((learned / "cost_point.json").read_text())
    for root, report in ((native, nr), (learned, lr)):
        for name in ("sampled_unit_counts.npy", "ensemble_fields.npz"):
            _require(sha256_file(root / name) == report["artifacts"][name], "saved sample changed: " + name)
    names = ("unit_ids", "unit_field_index", "field_ids", "unit_count", "observed_unit_counts", "observed_field_counts", "stimulus")
    with np.load(native / "ensemble_fields.npz", allow_pickle=False) as nf, np.load(learned / "ensemble_fields.npz", allow_pickle=False) as lf:
        validate_alignment(nr, lr, {k: nf[k] for k in names}, {k: lf[k] for k in names})
    totals = []
    for root in (native, learned):
        counts = np.load(root / "sampled_unit_counts.npy", mmap_mode="r", allow_pickle=False)
        _require(counts.shape == (16, 100, 100000), "complete saved count tensor required")
        totals.append(horizon_totals(counts, (4, 16, 100)))
        del counts
    rows, sensitivity, kernels = [], [], {}
    for bins in (4, 16, 100):
        summary, deleted, gram = describe_totals(totals[0][bins], totals[1][bins])
        rows.append(dict(horizon_ms=5 * bins, unit_count=100000, ordered_off_diagonal_unit_pairs=100000 * 99999,
                         native_draws=16, learned_draws=16, paired_differences_per_law=8, **summary))
        sensitivity.extend(dict(horizon_ms=5 * bins, **row) for row in deleted)
        kernels.update({f"bins{bins}_{law}": value for law, value in gram.items()})
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "unit_covariance.csv", rows)
    _write_csv(output / "delete_pair_sensitivity.csv", sensitivity)
    np.savez(output / "covariance_kernels.npz", **kernels)
    report = dict(status="DESCRIBED_FULL_UNIT_COVARIANCE_DEVELOPMENT", case_id=lr["configuration"]["case_id"],
        training_seed=41, native_initial_noise_blocks=1, connectivity_realizations=1,
        boundary_tick=10000, bin_ms=5., horizons_ms=[20, 80, 500], units="spike_count^4",
        statistic="mean squared off-diagonal covariance difference of horizon-total unit counts",
        correction="independent adjacent paired differences; distinct-pair U-statistic; signed estimates retained",
        draw_pairs=[[i, i + 1] for i in range(0, 16, 2)], covariance_matrix_allocated=False,
        independent_cross_ensemble_assumption="separately generated native future-noise and learned sampling streams",
        native_conditioning=nr["conditional_scope"], learned_conditioning=lr["conditional_scope"],
        conditioning_matched=False, uncertainty="delete-one-pair sensitivity only; no CI, p-value or FDR",
        scope="one development case; integrated counts only; temporal covariance not evaluated",
        interpretation="cross-law difference is descriptive; does not identify model error or qualify joint fidelity",
        simulation_performed=False, generation_performed=False, training_performed=False,
        formal_test_accessed=False, scientific_claim_qualified=False, matched_fidelity_qualified=False,
        source_binding=lr["source_binding"], checkpoint_sha256=lr["checkpoint_sha256"],
        input_report_sha256={"native": sha256_file(native / "analysis.json"), "learned": sha256_file(learned / "cost_point.json")},
        input_sample_sha256={"native": nr["artifacts"], "learned": lr["artifacts"]},
        summary=rows, artifacts={name: sha256_file(output / name) for name in
                               ("unit_covariance.csv", "delete_pair_sensitivity.csv", "covariance_kernels.npz")},
        analysis_seconds=time.perf_counter() - started, implementation_sha256=sha256_file(Path(__file__)))
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--native', type=Path)
    mode.add_argument('--retained-cell', type=Path, action='append', default=[])
    parser.add_argument('--previous-analysis', type=Path, action='append', default=[])
    parser.add_argument('--learned', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.native or args.retained_cell or args.previous_analysis):
        parser.error('native inputs or retained cells/previous analyses required')
    if bool(args.native) != bool(args.learned):
        parser.error('native comparison requires learned input; retained cells use no learned input')
    if args.native and args.previous_analysis:
        parser.error('previous retained analyses apply only to retained cells')
    result = (analyze(args.native, args.learned, args.output) if args.native
              else analyze_retained_cells(args.retained_cell, args.output, previous_analyses=args.previous_analysis))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

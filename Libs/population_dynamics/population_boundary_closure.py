"""Attach complete population boundaries to the archived residual-history probe.

This inference-only development diagnostic reuses every original episode,
boundary and target. It does not fit neural parameters or generate trajectories.
Formal state selection remains closed until its tolerances are qualified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .closure_selection import assess_state
from .field_history_probe import validate_config as validate_history_config
from .native_renewal_data import NativeRenewalEpisode
from .population_trajectory_evaluation import _load_model
from .provenance import sha256_file
from .trajectory_scores import field_pairs


def validate_config(config):
    expected = dict(schema="neural_population_boundary_closure_v1", scope="development_only",
        training_seeds=[41, 42, 43], fit_blocks=list(range(14)), selection_blocks=list(range(14, 18)),
        expected_episodes=72, expected_units=100000, expected_fields=128,
        origins_ms=[1000, 3000, 5000],
        boundary_features="latent_mean_variance_and_every_unit_ready_in_ticks_over_minimum_interval",
        added_history="archived_1000ms_field_history_in_50ms_bins",
        targets="reuse_complete_archived_50ms_500ms_mean_and_p1_variograms",
        control_conditioning="template_origin_cells_with_fixed_known_control_programs",
        witness="shared_linear_ridge_with_unpenalized_cell_effects", penalty=.01,
        formal_tolerances=None, state_selection_enabled=False, inference_only=True)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("registered complete inference-only development witness required")


def _read(path):
    return json.loads(path.read_text())


def load_history(root: Path, acquisition: Path, fields: Path, config):
    """Bind the existing measured targets instead of building another target set."""
    delivery = _read(root / "delivery.json")
    if (delivery["status"] != "ARCHIVED_INDEPENDENTLY_REPLAYED_DEVELOPMENT_PROBE"
            or delivery["calibration_outcomes_accessed"] is not False
            or delivery["test_outcomes_accessed"] is not False
            or sha256_file(root / "payload_manifest.json") != delivery["payload_manifest_sha256"]):
        raise ValueError("accepted archived development history probe required")
    manifest = _read(root / "payload_manifest.json")["files"]
    index = {row["path"]: row for row in manifest}
    if len(index) != len(manifest):
        raise ValueError("duplicate history delivery paths")
    for name in ("features/features.json", "features/features.npz"):
        row = index[name]
        if (root / name).stat().st_size != row["bytes"] or sha256_file(root / name) != row["sha256"]:
            raise ValueError("archived history features changed")
    report = _read(root / "features/features.json")
    validate_history_config(report["config"])
    if (report["status"] != "BUILT_DEVELOPMENT"
            or report["source_design_sha256"] != sha256_file(acquisition / "design.json")
            or report["source_dataset_sha256"] != sha256_file(fields / "dataset.json")
            or report["unit_count"] != config["expected_units"]
            or report["field_count"] != config["expected_fields"]
            or report["config"]["fit_blocks"] != config["fit_blocks"]
            or report["config"]["selection_blocks"] != config["selection_blocks"]
            or report["config"]["origins_ms"] != config["origins_ms"]
            or report["config"]["horizons_ms"] != [50, 500]
            or report["config"]["history_ms"] != 1000
            or report["config"]["history_bin_ms"] != 50
            or report["config"]["temporal_lag_ms"] != 5
            or report["source_bin_ms"] != 5
            or report["calibration_outcomes_accessed"] is not False
            or report["test_outcomes_accessed"] is not False
            or report["features_sha256"] != sha256_file(root / "features/features.npz")):
        raise ValueError("complete source-bound historical feature protocol required")
    with np.load(root / "features/features.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    rows = report["rows"]
    expected = {(b, f"familiar_{t}", origin) for b in range(18) for t in range(4)
                for origin in config["origins_ms"]}
    if (len(rows) != 216 or {(r["block"], r["template_id"], r["origin_ms"]) for r in rows} != expected
            or arrays["recent"].shape != (216, 128)
            or arrays["recent_and_history"].shape != (216, 2688)
            or arrays["targets"].shape != (216, 1920)
            or arrays["blocks"].shape != (216,) or arrays["cells"].shape != (216,)
            or not np.array_equal(arrays["blocks"], [r["block"] for r in rows])
            or not np.array_equal(arrays["cells"], [r["cell"] for r in rows])
            or not np.array_equal(arrays["recent"], arrays["recent_and_history"][:, :128])
            or not all(np.isfinite(arrays[k]).all() for k in ("recent", "recent_and_history", "targets"))):
        raise ValueError("all 216 identified history boundaries and 1920 targets required")
    for row in rows:
        template = int(row["template_id"].removeprefix("familiar_"))
        if (row["case_id"] != f"train_block_{row['block']:03d}_familiar_{template}"
                or row["cell"] != template * 3 + config["origins_ms"].index(row["origin_ms"])):
            raise ValueError("history case/cell identities differ from registered boundaries")
    source_bindings = report["source_bindings"]
    if (len(source_bindings) != 72 or {r["case_id"] for r in source_bindings}
            != {r["case_id"] for r in rows}):
        raise ValueError("all 72 unique historical episode bindings required")
    if arrays["field_ids"].shape != (128,) or len(set(map(str, arrays["field_ids"]))) != 128:
        raise ValueError("complete unique source field order required")
    definitions = [json.loads(str(value)) for value in arrays["field_ids"]]
    pairs = field_pairs([r[0] for r in definitions], [r[2] for r in definitions], [r[1] for r in definitions])
    for family in ("spatial", "cross_population"):
        if not np.array_equal(arrays[family + "_pairs"], pairs[family]):
            raise ValueError("archived source field target pairs changed")
    dimensions = dict(mean_rate=128, spatial_variogram=len(pairs["spatial"]),
                      cross_population_variogram=len(pairs["cross_population"]), temporal_variogram=128)
    expected_slices = [(family, horizon, width) for horizon in (50, 500)
                       for family, width in dimensions.items()]
    if len(report["target_slices"]) != len(expected_slices):
        raise ValueError("complete named target slices required")
    offset, names = 0, []
    for spec, (family, horizon, width) in zip(report["target_slices"], expected_slices):
        if (spec["start"] != offset or spec["stop"] != offset + width
                or spec["family"] != family or spec["horizon_ms"] != horizon
                or spec["statistic_units"] != "Hz" or spec["mse_units"] != "Hz^2"):
            raise ValueError("complete contiguous target slices required")
        names.extend(f"{spec['family']}_{spec['horizon_ms']}ms_{j}" for j in range(spec["stop"]-offset))
        offset = spec["stop"]
    if offset != arrays["targets"].shape[1] or len(set(names)) != len(names):
        raise ValueError("complete unique target definitions required")
    return report, arrays, names


def boundary_vector(state, model):
    """Keep both posterior moments and every source-ordered availability value."""
    if (state.unit_ids != model.unit_ids or state.field_names != model.field_names
            or state.bin_seconds != model.bin_seconds):
        raise ValueError("boundary/model source identities differ")
    interval = model.minimum_interval_ticks.detach().cpu().numpy().astype(np.float64)
    if (model.minimum_interval_ticks.dtype not in (torch.int32, torch.int64)
            or interval.shape != (len(model.unit_ids),) or not np.isfinite(interval).all() or np.any(interval <= 0)):
        raise ValueError("positive source minimum interval for every ordered unit required")
    model._ready(state.ready_in_ticks, 1)
    mean, variance = (x.detach().cpu().numpy().astype(np.float64) for x in (state.mean, state.variance))
    if (mean.shape != variance.shape or mean.shape != (1, len(model.field_names), model.latent_channels)
            or not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0)):
        raise ValueError("complete finite boundary posterior moments required")
    ready = state.ready_in_ticks.detach().cpu().numpy().astype(np.float64)
    return np.concatenate((mean.ravel(), variance.ravel(), (ready / interval).ravel()))


def extract_boundaries(model, native, origins_ms):
    """Read one observed window, then infer each strictly causal prefix."""
    ticks = []
    for origin in origins_ms:
        value = origin / native.dt_ms
        if (not np.isfinite(value) or value != int(value) or value <= 0
                or value > native.duration_ticks or int(value) % native.bin_ticks):
            raise ValueError("boundary origins must align with the complete native clock")
        ticks.append(int(value))
    if not ticks or len(set(ticks)) != len(ticks):
        raise ValueError("distinct causal boundary origins required")
    if (tuple(map(int, native.unit_ids)) != model.unit_ids
            or tuple(map(str, native.field_ids)) != model.field_names
            or not np.array_equal(native.unit_field_index, model.unit_field_index.cpu().numpy())
            or not np.array_equal(native.minimum_interval_ticks, model.minimum_interval_ticks.cpu().numpy())
            or native.dt_ms / 1000 != model.tick_seconds or native.bin_ms / 1000 != model.bin_seconds):
        raise ValueError("boundary/native source, clock or observation map changed")
    y, u = native.window(0, max(ticks))
    vectors, timings = [], []
    for tick in ticks:
        bins = tick // native.bin_ticks
        started = time.perf_counter()
        with torch.no_grad():
            state = model.infer_state(dict(counts=y[None, :bins], stimulus=u[None, :bins],
                                           ready_in_ticks=native.ready_at(tick)[None]))
        vectors.append(boundary_vector(state, model))
        timings.append(time.perf_counter() - started)
    return vectors, timings


def witness_from_boundaries(boundaries, report, arrays, names, config, *, witness_kind="linear"):
    """Use the original independent block identities; no seed-level inference."""
    states = np.asarray(boundaries, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] != len(report["rows"]) or not np.isfinite(states).all():
        raise ValueError("complete finite row-aligned boundary matrix required")
    fit = np.isin(arrays["blocks"], config["fit_blocks"])
    selection = np.isin(arrays["blocks"], config["selection_blocks"])
    if not np.all(fit ^ selection):
        raise ValueError("all boundaries must belong to disjoint registered blocks")
    cells = arrays["cells"]
    if cells.dtype.kind not in "iu" or np.any(cells < 0) or np.any(cells >= 12):
        raise ValueError("registered template/origin control cells required")
    controls = np.eye(12)[cells]
    arguments = {}
    for prefix, mask in (("fit", fit), ("selection", selection)):
        for name, values in (("state", states), ("history", arrays["recent_and_history"][:, 128:]),
                             ("controls", controls), ("targets", arrays["targets"]),
                             ("blocks", arrays["blocks"]), ("cells", cells)):
            arguments[prefix + "_" + name] = values[mask]
    return assess_state(**arguments, penalty=config["penalty"], target_names=names, witness_kind=witness_kind)


def run(acquisition: Path, fields: Path, history: Path, campaign: Path, config_path: Path, output: Path):
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    feature_report, arrays, names = load_history(history, acquisition, fields, config)
    design = _read(acquisition / "design.json")
    cases = {case["case_id"]: case for case in design["cases"]}
    if len(cases) != config["expected_episodes"] or len(cases) != len(design["cases"]):
        raise ValueError("complete native development cohort required")
    programs = {}
    for row in feature_report["rows"]:
        case = cases[row["case_id"]]
        if (case["split"] != "train" or case["block"] != row["block"]
                or case["template_id"] != row["template_id"]
                or row["cell"] != int(row["template_id"].removeprefix("familiar_"))*3
                    + config["origins_ms"].index(row["origin_ms"])):
            raise ValueError("history rows differ from native condition identities")
        control = case["settings"]["pulse_control"]
        key = row["template_id"]
        if key in programs and control != programs[key]:
            raise ValueError("template does not identify one fixed known control program")
        programs[key] = control
    output.mkdir(parents=True, exist_ok=False)
    receipts, family_rows = [], []
    torch.set_num_threads(1)
    for seed in config["training_seeds"]:
        started = time.perf_counter()
        training = campaign / f"seed_{seed}/training"
        first = next(iter(cases))
        native = NativeRenewalEpisode(acquisition / first, fields / first)
        model, binding, checkpoint_digest = _load_model(training, acquisition, fields, "population",
                                                       torch.device("cpu"), native)
        if binding["seed"] != seed or len(model.unit_ids) != config["expected_units"]:
            raise ValueError("registered complete population checkpoint required")
        vectors = [None] * len(feature_report["rows"])
        timings = []
        for identity in cases:
            indices = [i for i, row in enumerate(feature_report["rows"]) if row["case_id"] == identity]
            source = next(r for r in feature_report["source_bindings"] if r["case_id"] == identity)
            if (source["field_summary_sha256"] != sha256_file(fields / identity / "summary.json")
                    or source["spatial_fields_sha256"] != sha256_file(fields / identity / "spatial_fields.npz")):
                raise ValueError("historical target source changed")
            native = NativeRenewalEpisode(acquisition / identity, fields / identity,
                expected_source=binding["native_source_identity"], expected_unit_ids=np.asarray(model.unit_ids),
                expected_unit_field_index=model.unit_field_index.numpy())
            values, elapsed = extract_boundaries(model, native,
                [feature_report["rows"][i]["origin_ms"] for i in indices])
            for i, value in zip(indices, values):
                vectors[i] = value
            timings.extend(elapsed)
            del values, native
        matrix = np.asarray(vectors)
        result = witness_from_boundaries(matrix, feature_report, arrays, names, config)
        cell = output / f"seed_{seed}"
        cell.mkdir()
        np.savez_compressed(cell / "boundaries.npz", boundaries=matrix, unit_ids=np.asarray(model.unit_ids),
                            field_ids=np.asarray(model.field_names), blocks=arrays["blocks"], cells=arrays["cells"],
                            case_ids=np.asarray([r["case_id"] for r in feature_report["rows"]]),
                            origins_ms=np.asarray([r["origin_ms"] for r in feature_report["rows"]]))
        result.update(training_seed=seed, checkpoint_sha256=checkpoint_digest,
            source_binding=binding, boundary_dimensions=matrix.shape[1], complete_unit_availability=True,
            initial_process_observation_rng_consumed=False, inference_seconds=sum(timings),
            elapsed_seconds=time.perf_counter()-started, target_slices=feature_report["target_slices"],
            boundary_artifact_sha256=sha256_file(cell / "boundaries.npz"), configuration=config,
            interpretation=config["interpretation"], selection_blocks_used_for_checkpoint_selection=True,
            residual_gain_proves_missing_state=False, state_sufficiency_proven=False,
            selection_enabled=False, state_selected=None, scientific_claim_qualified=False)
        (cell / "witness.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        receipts.append(dict(seed=seed, boundaries=len(matrix), witness_sha256=sha256_file(cell / "witness.json")))
        for block in result["source_block_rows"]:
            for spec in feature_report["target_slices"]:
                take = slice(spec["start"], spec["stop"])
                family_rows.append(dict(training_seed=seed, block=block["block"],
                    family=spec["family"], horizon_ms=spec["horizon_ms"], rows=block["rows"],
                    dimensions=spec["stop"] - spec["start"],
                    state_mse=float(np.mean(block["state_mse"][take])),
                    augmented_mse=float(np.mean(block["augmented_mse"][take])),
                    history_gain=float(np.mean(block["history_gain"][take])),
                    score_units="per_target_fit_scale_normalized_squared_error"))
        print(json.dumps(receipts[-1]), flush=True)
        del model, vectors, matrix
    with (output / "errors_by_block.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    summary = dict(status="COMPLETE_THREE_SEED_BOUNDARY_HISTORY_DIAGNOSTIC", seed_receipts=receipts,
        config=config, history_feature_report_sha256=sha256_file(history / "features/features.json"),
        history_delivery_sha256=sha256_file(history / "delivery.json"),
        independent_selection_blocks=4, boundaries_per_seed=216, all_condition_failures_retained=True,
        selection_blocks_used_for_checkpoint_selection=True, state_sufficiency_proven=False,
        residual_gain_proves_missing_state=False, training_or_generation_repeated=False,
        interpretation=config["interpretation"],
        formal_tolerances_qualified=False, state_selection_enabled=False, scientific_claim_qualified=False,
        source_fidelity_qualified=False, calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def _bound_archive_file(root, index, name):
    path = root / name
    row = index[name]
    if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise ValueError(f"archived boundary input changed: {name}")
    return path


def load_cached_boundary(root, index, seed, report, arrays, names, config):
    """Admit an accepted complete boundary without opening a neural checkpoint."""
    prefix = f"analysis/seed_{seed}/"
    witness = _read(_bound_archive_file(root, index, prefix + "witness.json"))
    path = _bound_archive_file(root, index, prefix + "boundaries.npz")
    validate_config(witness["configuration"])
    if (witness["training_seed"] != seed or witness["target_names"] != names
            or witness["target_slices"] != report["target_slices"]
            or witness["boundary_artifact_sha256"] != index[prefix + "boundaries.npz"]["sha256"]
            or witness["control_conditioning"] != "fixed_factorial_cells_shared_linear_state"
            or witness["source_binding"]["seed"] != seed
            or witness["source_binding"]["acquisition_design_sha256"] != report["source_design_sha256"]
            or witness["source_binding"]["dataset_sha256"] != report["source_dataset_sha256"]
            or witness["selection_enabled"] is not False
            or witness["scientific_claim_qualified"] is not False):
        raise ValueError("accepted source-bound linear witness required")
    with np.load(path, allow_pickle=False) as data:
        states = data["boundaries"]
        units, fields = data["unit_ids"], data["field_ids"]
        expected_width = config["expected_units"] + 2 * config["expected_fields"] * witness["source_binding"]["config"]["latent_channels"]
        expected_rows = dict(blocks=arrays["blocks"], cells=arrays["cells"],
            case_ids=np.asarray([row["case_id"] for row in report["rows"]]),
            origins_ms=np.asarray([row["origin_ms"] for row in report["rows"]]))
        if (any(not np.array_equal(data[key], value) for key, value in expected_rows.items())
                or units.shape != (config["expected_units"],) or units.dtype.kind not in "iu"
                or len(np.unique(units)) != len(units)
                or not np.array_equal(fields, arrays["field_ids"])
                or states.shape != (len(report["rows"]), expected_width)
                or states.dtype != np.float64 or not np.isfinite(states).all()
                or witness["boundary_dimensions"] != expected_width):
            raise ValueError("complete source-ordered cached boundaries and row identities required")
    return states, units, witness


def compare_readout_families(seed, linear, nonlinear, specs):
    """Keep each block and target family; do not infer independence from seeds."""
    if (linear["target_names"] != nonlinear["target_names"]
            or linear["training_target_scale"] != nonlinear["training_target_scale"]
            or linear["defined_targets"] != nonlinear["defined_targets"]):
        raise ValueError("cached readouts must share targets and fitting-only normalization")
    left, right = linear["source_block_rows"], nonlinear["source_block_rows"]
    if [x["block"] for x in left] != [x["block"] for x in right]:
        raise ValueError("cached readouts must share all independent selection blocks")
    rows = []
    for old, current in zip(left, right):
        if old["rows"] != current["rows"]:
            raise ValueError("cached readouts must share every selection row")
        for spec in specs:
            take = slice(spec["start"], spec["stop"])
            row = dict(training_seed=seed, block=old["block"], family=spec["family"],
                       horizon_ms=spec["horizon_ms"], rows=old["rows"], dimensions=spec["stop"]-spec["start"])
            for kind, result in (("linear", old), ("rbf", current)):
                for key in ("state_mse", "augmented_mse", "history_gain"):
                    values = np.asarray(result[key], dtype=float)
                    if values.shape != (len(linear["target_names"]),) or not np.isfinite(values).all():
                        raise ValueError("complete finite per-target readout errors required")
                    row[kind + "_" + key] = float(values[take].mean())
            rows.append(row)
    return rows


def _write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cached(acquisition, fields, history, boundary_run, config_path, output):
    """Fit only two RBF readouts per seed; reuse the archived linear results."""
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    delivery = _read(boundary_run / "delivery.json")
    if (delivery["status"] != "PASS" or delivery["exit_code"] != 0
            or delivery["training_seeds"] != config["training_seeds"]
            or delivery["mode"] != "inference_only"
            or delivery["calibration_or_test_accessed"] is not False):
        raise ValueError("accepted original three-seed boundary delivery required")
    index = {row["path"]: row for row in delivery["files"]}
    if len(index) != len(delivery["files"]):
        raise ValueError("duplicate archived boundary input paths")
    original = _read(_bound_archive_file(boundary_run, index, "analysis/analysis.json"))
    validate_config(original["config"])
    if (original["history_delivery_sha256"] != sha256_file(history / "delivery.json")
            or original["history_feature_report_sha256"] != sha256_file(history / "features/features.json")):
        raise ValueError("cached boundaries must reuse their original history targets")
    report, arrays, names = load_history(history, acquisition, fields, config)
    output.mkdir(parents=True, exist_ok=False)
    receipts, comparisons, reference_units = [], [], None
    for seed in config["training_seeds"]:
        started = time.perf_counter()
        states, units, linear = load_cached_boundary(boundary_run, index, seed, report, arrays, names, config)
        if reference_units is not None and not np.array_equal(reference_units, units):
            raise ValueError("all three checkpoints must share source unit order")
        reference_units = units
        result = witness_from_boundaries(states, report, arrays, names, config, witness_kind="rbf")
        comparisons.extend(compare_readout_families(seed, linear, result, report["target_slices"]))
        result.update(training_seed=seed, boundary_artifact_sha256=linear["boundary_artifact_sha256"],
            checkpoint_sha256=linear["checkpoint_sha256"], boundary_dimensions=states.shape[1],
            target_slices=report["target_slices"],
            archived_linear_witness_sha256=index[f"analysis/seed_{seed}/witness.json"]["sha256"],
            selection_blocks_used_for_checkpoint_selection=True,
            state_selection_enabled=False, state_selected=None, scientific_claim_qualified=False,
            neural_training_repeated=False, boundary_extraction_repeated=False, trajectories_generated=0,
            elapsed_seconds=time.perf_counter()-started)
        cell = output / f"seed_{seed}"
        cell.mkdir()
        (cell / "witness.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        receipt = dict(seed=seed, witness_sha256=sha256_file(cell / "witness.json"),
                       elapsed_seconds=result["elapsed_seconds"])
        for kind, witness in (("linear", linear), ("rbf", result)):
            for arm in ("state", "augmented"):
                receipt[kind + "_" + arm] = witness[arm + "_predictor_diagnostic"]
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
        del states, linear, result
    _write_csv(output / "errors_by_block.csv", comparisons)
    summaries = []
    for spec in report["target_slices"]:
        matching = [row for row in comparisons if row["family"] == spec["family"] and row["horizon_ms"] == spec["horizon_ms"]]
        row = dict(family=spec["family"], horizon_ms=spec["horizon_ms"], seed_block_rows=len(matching),
                   independent_selection_blocks=len(config["selection_blocks"]))
        for kind in ("linear", "rbf"):
            for key in ("state_mse", "augmented_mse", "history_gain"):
                column = kind + "_" + key
                row[column] = float(np.mean([item[column] for item in matching]))
        row.update(state_error_improved=row["rbf_state_mse"] < row["linear_state_mse"],
                   history_gain_reduced=row["rbf_history_gain"] < row["linear_history_gain"],
                   positive_rbf_history_gain=row["rbf_history_gain"] > 0)
        summaries.append(row)
    _write_csv(output / "family_summary.csv", summaries)
    summary = dict(status="COMPLETE_CACHED_BOUNDARY_READOUT_DIAGNOSTIC", config=config,
        witness_kind="rbf", penalty=config["penalty"], regularization_selected=False,
        kernel_bandwidth="sqrt(number_of_fit_active_features)", matched_effective_complexity=False,
        boundary_delivery_sha256=sha256_file(boundary_run / "delivery.json"),
        history_delivery_sha256=sha256_file(history / "delivery.json"), seed_receipts=receipts,
        complete_episodes=72, boundaries_per_seed=216, target_dimensions=1920,
        fitting_rows_per_seed=168, selection_rows_per_seed=48, independent_selection_blocks=4,
        family_block_rows=len(comparisons), score_units="per_target_fit_scale_normalized_squared_error",
        neural_training_repeated=False, boundary_extraction_repeated=False, trajectories_generated=0,
        linear_regression_repeated=False, all_condition_failures_retained=True,
        selection_blocks_used_for_checkpoint_selection=True,
        formal_tolerances_qualified=False, state_selection_enabled=False, state_sufficiency_proven=False,
        residual_gain_proves_missing_state=False, scientific_claim_qualified=False,
        calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        interpretation="Lower state-only error with reduced history gain is consistent with a decoder limitation. "
            "Persistent gain under an adequate readout motivates state investigation but proves no missing information. "
            "Worse state-only prediction makes this probe inconclusive. Equal penalties do not match effective "
            "complexity across kernels; all unit-readiness features are retained. No closure certification follows.",
        completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "history", "config"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("campaign", "output"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cached-boundary-run", type=Path)
    args = parser.parse_args()
    if args.cached_boundary_run is not None:
        if args.campaign is not None or args.validate_only or args.output is None:
            parser.error("cached readout requires --output and excludes --campaign/--validate-only")
        print(json.dumps(run_cached(args.acquisition, args.fields, args.history,
                                    args.cached_boundary_run, args.config, args.output)))
        return
    if args.validate_only:
        config = yaml.safe_load(args.config.read_text())
        validate_config(config)
        report, arrays, names = load_history(args.history, args.acquisition, args.fields, config)
        print(json.dumps(dict(status="REGISTERED_HISTORY_INPUTS_VALIDATED", boundaries=len(report["rows"]),
                              targets=len(names), scientific_claim_qualified=False)))
        return
    if args.campaign is None or args.output is None:
        parser.error("--campaign and --output are required for boundary inference")
    print(json.dumps(run(args.acquisition, args.fields, args.history, args.campaign, args.config, args.output)))


if __name__ == "__main__":
    main()

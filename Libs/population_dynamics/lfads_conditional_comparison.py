"""Compare one fitting-derived conditional-mean arm with its archived AR reference."""
from __future__ import annotations

import argparse
import csv
import json
from math import fsum, isfinite
from pathlib import Path

import numpy as np

from .lfads_conditional_prior import load_mean_prior
from .lfads_forecast_analysis import (
    _delivered_files, _mean, _readout_inputs, _replayed_forecast,
    field_mean_error_components, observation_noise_components,
)
from .provenance import freeze_json, sha256_file
from .trajectory_forecast_analysis import _write_rows

REFERENCE = "fitted_zero_mean_ar_prior"
CANDIDATE = "template_time_controller_mean_prior"
ARMS = (REFERENCE, CANDIDATE)
BLOCKS = (14, 15, 16, 17)
ORIGINS = (1000., 3000., 5000.)
PRIMARY = ("conditional_empirical_mean_mse_hz2", "absolute_global_mean_rate_bias_hz")
GUARDS = ("variogram_temporal_lag_4", "refractory_count_violation_1bin",
          "refractory_count_violation_2bin", "refractory_count_violation_4bin")
METRICS = PRIMARY + GUARDS
IDENTITY = ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms", "arm")


def verify_paired_paths(reference, candidate):
    pairs = [(name, name) for name in (
        "observed_field_counts", "lfads_initial_latent_states", "lfads_historical_controller_inputs",
        "lfads_boundary_generator_states")]
    pairs.append(("lfads_future_controller_inputs", "lfads_base_future_controller_inputs"))
    for left_name, right_name in pairs:
        left, right = reference[left_name], candidate[right_name]
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError("PAIRED_CAUSAL_PATH_CHANGED:" + left_name)


def conditional_metrics(samples, expected, truth, units, bin_seconds):
    weights = units / units.sum()
    metrics = field_mean_error_components(samples, truth, units, bin_seconds)
    metrics.update({"conditional_" + k: v for k, v in
        field_mean_error_components(expected, truth, units, bin_seconds).items()})
    metrics.update(observation_noise_components(samples, expected, truth, units, bin_seconds))
    result = {key: float(value @ weights) for key, value in metrics.items()}
    result["absolute_global_mean_rate_bias_hz"] = abs(result["conditional_signed_time_mean_bias_hz"])
    return result, metrics


def compare_per_block(origin_rows):
    indexed = {(arm, block): {} for arm in ARMS for block in BLOCKS}
    for row in origin_rows:
        arm, block = row["arm"], row["block"]
        if arm not in ARMS or block not in BLOCKS or row["horizon_ms"] != 500:
            raise ValueError("Unexpected arm, block or routing horizon")
        key = (row["case_id"], row["origin_ms"])
        bucket = indexed[arm, block]
        if key in bucket:
            raise ValueError("Duplicate routing row")
        values = {metric: float(row[metric]) for metric in METRICS}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite routing metric")
        bucket[key] = values
    blocks = []
    for block in BLOCKS:
        expected = {(f"train_block_{block:03d}_familiar_{template}", origin)
                    for template in range(4) for origin in ORIGINS}
        if any(set(indexed[arm, block]) != expected for arm in ARMS):
            raise ValueError("Incomplete or unpaired registered origin set")
        means = {arm: {metric: fsum(indexed[arm, block][key][metric] for key in sorted(expected)) / 12
                       for metric in METRICS} for arm in ARMS}
        passed = {metric: (means[CANDIDATE][metric] < means[REFERENCE][metric] if metric in PRIMARY
                          else means[CANDIDATE][metric] <= means[REFERENCE][metric]) for metric in METRICS}
        blocks.append(dict(block=block, reference=means[REFERENCE], candidate=means[CANDIDATE],
            deltas={metric: means[CANDIDATE][metric] - means[REFERENCE][metric] for metric in METRICS},
            metric_pass=passed, route="PASS" if all(passed.values()) else "STOP"))
    return dict(reference_arm=REFERENCE, candidate_arm=CANDIDATE, horizon_ms=500, blocks=blocks,
        route="PASS" if all(row["route"] == "PASS" for row in blocks) else "STOP")


def _reference_metrics(root, evaluation, replay, replay_path, seed):
    """Reuse the completed native-readout reconstruction; never redraw the reference."""
    receipt = json.loads((root / "delivery.json").read_text())
    analysis_path = root / "analysis/analysis.json"
    entries = {row["path"]: row for row in receipt["files"]}
    if (receipt["status"] != "RECONSTRUCTED_REPLAYED_CONDITIONAL_MEANS" or receipt["exit_code"] != 0
            or sha256_file(analysis_path) != entries["analysis/analysis.json"]["sha256"]):
        raise ValueError("COMPLETED_REFERENCE_READOUT_REQUIRED")
    analysis = json.loads(analysis_path.read_text())
    binding = analysis["arm_bindings"]["fitted_prior"]
    if (binding["evaluation_sha256"] != replay["evaluation_sha256"]
            or binding["replay_sha256"] != sha256_file(replay_path)
            or binding["scores_sha256"] != evaluation["scores_sha256"]
            or binding["prior_refit"] != evaluation["prior_refit"]
            or analysis["training_binding"] != evaluation["training_binding"]
            or analysis["configuration"] != evaluation["configuration"]
            or analysis["conditional_mean_reconstruction"]["checkpoint_artifacts"] != evaluation["selected_model_artifacts"]
            or analysis["conditional_mean_reconstruction"]["implementation_sha256"] !=
                sha256_file(Path(__file__).with_name("lfads_readout.py"))):
        raise ValueError("REFERENCE_READOUT_BINDING_CHANGED")
    path = root / "analysis/field_error_by_origin.csv"
    if sha256_file(path) != analysis["artifacts"][path.name]:
        raise ValueError("REFERENCE_READOUT_TABLE_CHANGED")
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] != "fitted_prior":
                continue
            key = (row["case_id"], float(row["origin_ms"]), float(row["horizon_ms"]))
            if key in rows or int(row["seed"]) != seed:
                raise ValueError("REFERENCE_READOUT_IDENTITIES_CHANGED")
            values = {key: float(value) for key, value in row.items() if key not in IDENTITY}
            values["absolute_global_mean_rate_bias_hz"] = abs(values["conditional_signed_time_mean_bias_hz"])
            rows[key] = values
    return rows, dict(delivery_sha256=sha256_file(root / "delivery.json"),
        analysis_sha256=sha256_file(analysis_path), table_sha256=sha256_file(path))


def compare(reference, candidate, reference_readout, mean, checkpoint, field_view, output):
    reference, candidate, mean, checkpoint, field_view, output = map(
        Path, (reference, candidate, mean, checkpoint, field_view, output))
    reference_readout = None if reference_readout is None else Path(reference_readout)
    if output.exists():
        raise FileExistsError(output)
    mean_record = json.loads((mean / "mean_prior.json").read_text())
    seed = mean_record["training_seed"]
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("REGISTERED_CONDITIONAL_MEAN_TRAINING_SEED_REQUIRED")
    left, left_replay, scale, left_rows = _replayed_forecast(reference / "forecasts", reference / "replay.json", seed)
    right, right_replay, right_scale, right_rows = _replayed_forecast(candidate / "forecasts", candidate / "replay.json", seed)
    matched = ("training_binding", "training_run_sha256", "selected_checkpoint", "selected_training_sha256",
               "selected_model_artifacts", "selected_epoch", "native_source_identity", "configuration",
               "inference_migration", "prior_refit")
    if (any(left[key] != right[key] for key in matched)
            or any(not np.array_equal(scale[key], right_scale[key]) for key in scale)
            or set(left_rows) != set(right_rows) or "conditional_prior" in left):
        raise ValueError("MATCHED_REFERENCE_AND_CONDITIONAL_ARM_REQUIRED")
    prior = left["prior_refit"]
    budget = left["effective_training_epoch_budget"]
    if (prior["fitted_parameters"] != 8 or prior["fitting_episodes"] != 56
            or prior["training_seed"] != seed or budget not in (20, 80)
            or prior["selected_epoch"] != left["selected_epoch"] + 1
            or right["effective_training_epoch_budget"] != budget):
        raise ValueError("REGISTERED_SEED_BUDGET_CHECKPOINT_REFERENCE_REQUIRED")
    _, mean_binding = load_mean_prior(mean, prior, training_seed=seed,
        network_budget_epochs=budget, selected_model_artifacts=left["selected_model_artifacts"])
    if right.get("conditional_prior") != mean_binding or right_replay.get("conditional_prior") != mean_binding:
        raise ValueError("CANDIDATE_MEAN_BINDING_CHANGED")
    protocol = mean_binding["comparison_protocol"]
    if (protocol["reference_arm"] != REFERENCE or protocol["candidate_arm"] != CANDIDATE
            or protocol["primary"]["metric"] != PRIMARY[0] or protocol["primary"]["horizon_ms"] != 500
            or protocol["routing_rule"]["horizon_ms"] != 500
            or protocol["routing_rule"]["each_selection_block_must_improve"] != list(PRIMARY)
            or protocol["routing_rule"]["each_selection_block_must_not_worsen"] != list(GUARDS)
            or protocol["routing_rule"]["block_aggregation"] != "equal_mean_over_all_12_registered_origins"
            or protocol["routing_rule"]["absolute_global_mean_rate_bias_hz"] !=
                "absolute_conditional_minus_observed_unit_weighted_global_mean_rate_per_origin_then_block_mean"):
        raise ValueError("PROSPECTIVE_ROUTING_PROTOCOL_CHANGED")
    archived = reference_binding = None
    if reference_readout is not None:
        archived, reference_binding = _reference_metrics(reference_readout, left, left_replay, reference / "replay.json", seed)
        if set(archived) != set(left_rows):
            raise ValueError("COMPLETE_REFERENCE_CONDITIONAL_METRICS_REQUIRED")
    readout, mapping = _readout_inputs(right, scale, checkpoint, field_view)
    units = scale["unit_count"]
    bin_seconds = right["training_binding"]["config"]["bin_ms"] / 1000
    all_rows, candidate_fields, reference_fields = [], [], []
    for block in BLOCKS:
        for template in range(4):
            case = f"train_block_{block:03d}_familiar_{template}"
            for origin in ORIGINS:
                name = f"{case}_origin_{origin:g}.npz"
                with np.load(reference / "forecasts" / name, allow_pickle=False) as ref, np.load(
                        candidate / "forecasts" / name, allow_pickle=False) as cand:
                    verify_paired_paths(ref, cand)
                    samples, truth = cand["sampled_field_counts"], cand["observed_field_counts"]
                    expected, _ = readout.expected_field_counts(cand["lfads_future_generator_states"], mapping, len(units))
                    if archived is None:
                        reference_expected, _ = readout.expected_field_counts(ref["lfads_future_generator_states"], mapping, len(units))
                        reference_samples = ref["sampled_field_counts"]
                        reference_truth = ref["observed_field_counts"]
                for horizon in right["configuration"]["horizons_ms"]:
                    bins = int(horizon / (bin_seconds * 1000))
                    key = (case, origin, horizon)
                    new_metrics, by_field = conditional_metrics(samples[:, :bins], expected[:, :bins], truth[:bins], units, bin_seconds)
                    if archived is None:
                        old_metrics, old_by_field = conditional_metrics(reference_samples[:, :bins],
                            reference_expected[:, :bins], reference_truth[:bins], units, bin_seconds)
                    else:
                        old_metrics = archived[key]
                    if new_metrics["conditional_observed_mean_rate_hz"] != old_metrics["conditional_observed_mean_rate_hz"]:
                        raise ValueError("PAIRED_OBSERVED_RATE_CHANGED")
                    identity = dict(seed=seed, block=block, case_id=case, template_id=f"familiar_{template}",
                                    origin_ms=origin, horizon_ms=horizon)
                    for arm, metrics, scores in ((REFERENCE, old_metrics, left_rows[key]),
                                                  (CANDIDATE, new_metrics, right_rows[key])):
                        guards = {metric: None if scores[metric] == "" else float(scores[metric]) for metric in GUARDS}
                        all_rows.append(dict(**identity, arm=arm, **metrics, **guards))
                    for index, field in enumerate(scale["field_ids"]):
                        candidate_fields.append(dict(**identity, arm=CANDIDATE, field_id=str(field), units=int(units[index]),
                            **{metric: float(values[index]) for metric, values in by_field.items()}))
                        if archived is None:
                            reference_fields.append(dict(**identity, arm=REFERENCE, field_id=str(field), units=int(units[index]),
                                **{metric: float(values[index]) for metric, values in old_by_field.items()}))
    decision = compare_per_block([row for row in all_rows if row["horizon_ms"] == 500])
    output.mkdir(parents=True, exist_ok=False)
    tables = {"conditional_by_origin.csv": all_rows, "candidate_field_errors.csv": candidate_fields}
    if archived is None:
        tables["reference_field_errors.csv"] = reference_fields
    differences = []
    for block in decision["blocks"]:
        for metric in METRICS:
            differences.append(dict(block=block["block"], horizon_ms=500, metric=metric,
                reference_arm=REFERENCE, candidate_arm=CANDIDATE, reference=block["reference"][metric],
                candidate=block["candidate"][metric], candidate_minus_reference=block["deltas"][metric],
                passed=block["metric_pass"][metric]))
    tables["paired_block_decisions.csv"] = differences
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    result = dict(status="DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_COMPARISON", **decision,
        training_seed=seed, network_budget_epochs=budget, training_seeds=1, independent_blocks=4, connectivity_instances=1,
        paired_origins=48, score_rows_per_arm=144, exact_paired_causal_paths=True,
        reference_readout_reused=reference_binding, reference_readout_reconstructed=(archived is None),
        reference_readout_implementation_sha256=sha256_file(Path(__file__).with_name("lfads_readout.py")),
        new_forecast_origins=0, new_spike_samples=0,
        reference_evaluation_sha256=left_replay["evaluation_sha256"],
        candidate_evaluation_sha256=right_replay["evaluation_sha256"], mean_binding=mean_binding,
        comparison_protocol=protocol, implementation_sha256=sha256_file(Path(__file__)),
        artifacts={name: sha256_file(output / name) for name in tables},
        scientific_claim_qualified=False, formal_test_accessed=False,
        uncertainty="Single-seed development routing; no inferential CI, p value, or cross-seed claim",
        next_action="continue_mean_variant_development" if decision["route"] == "PASS" else
                    "stop_mean_only_route_without_filling_other_five_cells")
    freeze_json(output / "analysis.json", result)
    return result


def _matrix_protocol(protocol, budget, seed):
    """Normalize only the registered single-cell/matrix scope difference."""
    if protocol.get("scope") == "one_development_causal_prior_contrast":
        if (budget, seed) != (20, 41) or protocol.get("training_seed") != 41 or protocol.get("network_budget_epochs") != 20:
            raise ValueError("LEGACY_PROTOCOL_ONLY_SUPPORTS_ORIGINAL_CELL")
        omitted = {"scope", "training_seed", "network_budget_epochs"}
    elif protocol.get("scope") == "registered_development_causal_prior_matrix":
        if protocol.get("training_seeds") != [41, 42, 43] or protocol.get("network_budget_epoch_ceilings") != [20, 80]:
            raise ValueError("REGISTERED_CONDITIONAL_MATRIX_PROTOCOL_REQUIRED")
        omitted = {"scope", "training_seeds", "network_budget_epoch_ceilings", "single_cell_protocol_sha256", "routing_reference"}
    else:
        raise ValueError("REGISTERED_CONDITIONAL_MATRIX_PROTOCOL_REQUIRED")
    return {key: value for key, value in protocol.items() if key not in omitted}


def _conditional_matrix_cell(budget, seed, archive, reference_delivery=None):
    """Read five small files; authenticate their existing archive bindings."""
    archive = Path(archive)
    delivery_path = archive / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    report_path = archive / "analysis/analysis.json"
    table_path = archive / "analysis/paired_block_decisions.csv"
    digests = {path: sha256_file(path) for path in (report_path, table_path)}
    _delivered_files(archive, delivery, digests)
    report = json.loads(report_path.read_text())
    expected = dict(status="DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_COMPARISON", training_seed=seed,
        training_seeds=1, independent_blocks=4, connectivity_instances=1, paired_origins=48,
        score_rows_per_arm=144, exact_paired_causal_paths=True, reference_arm=REFERENCE, candidate_arm=CANDIDATE,
        horizon_ms=500, scientific_claim_qualified=False, formal_test_accessed=False)
    if (delivery.get("status") != expected["status"] or delivery.get("exit_code") != 0
            or any(report.get(key) != value for key, value in expected.items())
            or report["artifacts"].get(table_path.name) != digests[table_path]
            or report.get("route") not in ("PASS", "STOP") or delivery.get("route") != report["route"]
            or report["mean_binding"]["comparison_protocol"] != report["comparison_protocol"]):
        raise ValueError("COMPLETE_BOUND_CONDITIONAL_COMPARISON_REQUIRED")
    protocol = report["comparison_protocol"]
    _matrix_protocol(protocol, budget, seed)
    if (report.get("network_budget_epochs", 20 if (budget, seed) == (20, 41) else None) != budget):
        raise ValueError("CONDITIONAL_COMPARISON_BUDGET_CHANGED")
    source_binding = delivery["reference_delivery"]
    parent_path = Path(reference_delivery) if reference_delivery is not None else Path(source_binding["path"])
    if sha256_file(parent_path) != source_binding["sha256"]:
        raise ValueError("CONDITIONAL_REFERENCE_DELIVERY_CHANGED")
    parent = json.loads(parent_path.read_text())
    if parent.get("status") == "COMPARED_PRIORS_DEVELOPMENT":
        prefix = "fitted_prior"
        parent_budget = {"budget_20": 20, "budget_80": 80}.get(parent.get("network_variant"))
        if "exit_code" in parent and parent["exit_code"] != 0:
            raise ValueError("COMPLETED_CONDITIONAL_REFERENCE_REQUIRED")
    elif parent.get("status") == "REPLAYED_STATIONARY_PRIOR_CANDIDATE" and parent.get("exit_code") == 0:
        prefix, parent_budget = "candidate", parent.get("network_budget_epochs")
    else:
        raise ValueError("COMPLETED_CONDITIONAL_REFERENCE_REQUIRED")
    evaluation_path = parent_path.parent / prefix / "forecasts/evaluation.json"
    evaluation_digest = sha256_file(evaluation_path)
    _delivered_files(parent_path.parent, parent, {evaluation_path: evaluation_digest})
    evaluation = json.loads(evaluation_path.read_text())
    training = evaluation["training_binding"]
    continuation = evaluation.get("training_continuation")
    prior = evaluation["prior_refit"]
    if (evaluation_digest != report["reference_evaluation_sha256"]
            or parent.get("training_seed") != seed or parent_budget != budget
            or evaluation.get("status") != "EVALUATED_DEVELOPMENT" or evaluation.get("training_seed") != seed
            or training.get("seed") != seed or evaluation.get("effective_training_epoch_budget") != budget
            or training["config"].get("maximum_epochs") != 20
            or evaluation.get("formal_test_accessed") is not False
            or prior.get("training_seed") != seed or prior.get("fitted_parameters") != 8 or prior.get("fitting_episodes") != 56
            or type(evaluation.get("selected_epoch")) is not int or type(evaluation.get("completed_training_epochs")) is not int
            or prior.get("selected_epoch") != evaluation["selected_epoch"]+1
            or not 0 <= evaluation["selected_epoch"] < evaluation["completed_training_epochs"] <= budget
            or (budget == 20 and continuation is not None)
            or (budget == 80 and (not isinstance(continuation, dict)
                or continuation.get("schema") != "neural_population_lfads_budget_continuation_v1"
                or continuation.get("total_epochs") != 80 or continuation.get("base_maximum_epochs") != 20
                or continuation.get("treatment") != "optimization_budget_only"))):
        raise ValueError("REFERENCE_SEED_BUDGET_OR_CHECKPOINT_PROVENANCE_CHANGED")
    source_keys = ("acquisition_design_sha256", "acquisition_ledger_sha256", "dataset_sha256",
                   "episode_table_sha256", "native_source_identity", "episode_bindings")
    source = {key: training[key] for key in source_keys}
    source.update(training_config=training["config"], native_evaluation_identity=evaluation["native_source_identity"],
                  forecast_configuration=evaluation["configuration"])
    if (training["config"].get("fit_blocks") != list(range(14))
            or training["config"].get("validation_blocks") != list(BLOCKS)):
        raise ValueError("FIXED_FITTING_AND_SELECTION_BLOCKS_REQUIRED")
    block_reports = {row["block"]: row for row in report["blocks"]}
    if len(report["blocks"]) != 4 or set(block_reports) != set(BLOCKS):
        raise ValueError("COMPLETE_CONDITIONAL_BLOCK_REPORT_REQUIRED")
    rows = {}
    with table_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["block"]), row["metric"]
            if (key in rows or key[0] not in BLOCKS or key[1] not in METRICS
                    or float(row["horizon_ms"]) != 500 or row["reference_arm"] != REFERENCE
                    or row["candidate_arm"] != CANDIDATE or row["passed"] not in ("True", "False")):
                raise ValueError("COMPLETE_UNIQUE_CONDITIONAL_DECISIONS_REQUIRED")
            ref, cand, delta = (float(row[name]) for name in ("reference", "candidate", "candidate_minus_reference"))
            passed = cand < ref if key[1] in PRIMARY else cand <= ref
            block = block_reports[key[0]]
            if (not all(isfinite(value) for value in (ref, cand, delta)) or delta != cand-ref
                    or (row["passed"] == "True") != passed
                    or block["reference"][key[1]] != ref or block["candidate"][key[1]] != cand
                    or block["deltas"][key[1]] != delta or block["metric_pass"][key[1]] is not passed):
                raise ValueError("CONDITIONAL_DECISION_VALUES_OR_ROUTING_CHANGED")
            rows[key] = dict(network_budget_epochs=budget, training_seed=seed, block=key[0], horizon_ms=500,
                metric=key[1], reference=ref, candidate=cand, candidate_minus_reference=delta, passed=passed,
                matched_episode_origins=12, reference_arm=REFERENCE, candidate_arm=CANDIDATE)
    if set(rows) != {(block, metric) for block in BLOCKS for metric in METRICS}:
        raise ValueError("ALL24_CONDITIONAL_DECISIONS_REQUIRED")
    for block, description in block_reports.items():
        route = "PASS" if all(rows[block, metric]["passed"] for metric in METRICS) else "STOP"
        if description["route"] != route:
            raise ValueError("CONDITIONAL_BLOCK_ROUTE_CHANGED")
    route = "PASS" if all(row["passed"] for row in rows.values()) else "STOP"
    if report["route"] != route:
        raise ValueError("CONDITIONAL_CELL_ROUTE_CHANGED")
    provenance = dict(network_budget_epochs=budget, training_seed=seed, archive=str(archive), route=route,
        comparison_delivery_sha256=sha256_file(delivery_path), analysis_sha256=digests[report_path],
        paired_decisions_sha256=digests[table_path], reference_delivery_path=source_binding["path"],
        reference_delivery_sha256=source_binding["sha256"], reference_evaluation_sha256=evaluation_digest,
        selected_epoch=evaluation["selected_epoch"], completed_training_epochs=evaluation["completed_training_epochs"],
        source_identity=source)
    return report, rows, provenance


def aggregate_matrix(cells, output, reference_deliveries=None):
    """Describe all six cells, averaging seeds within each shared block first."""
    expected = {(budget, seed) for budget in (20, 80) for seed in (41, 42, 43)}
    if (len(cells) != 6 or any(type(budget) is not int or type(seed) is not int for budget, seed, _ in cells)
            or {(budget, seed) for budget, seed, _ in cells} != expected):
        raise ValueError("EXACT_SIX_REGISTERED_CONDITIONAL_MATRIX_CELLS_REQUIRED")
    references = {} if reference_deliveries is None else dict(reference_deliveries)
    if not set(references) <= expected:
        raise ValueError("UNKNOWN_MATRIX_REFERENCE_CELL")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    loaded = {}
    for budget, seed, archive in sorted(cells):
        loaded[budget, seed] = _conditional_matrix_cell(budget, seed, archive, references.get((budget, seed)))
    first, _, first_provenance = loaded[20, 41]
    science = _matrix_protocol(first["comparison_protocol"], 20, 41)
    all_rows, inputs = [], []
    for (budget, seed), (report, rows, provenance) in loaded.items():
        protocol = report["comparison_protocol"]
        if (_matrix_protocol(protocol, budget, seed) != science
                or provenance["source_identity"] != first_provenance["source_identity"]):
            raise ValueError("SHARED_SOURCE_OR_CONDITIONAL_SCIENTIFIC_PROTOCOL_CHANGED")
        if protocol["scope"] == "registered_development_causal_prior_matrix":
            routing = protocol.get("routing_reference", {})
            if (protocol.get("single_cell_protocol_sha256") != first["mean_binding"]["comparison_protocol_sha256"]
                    or routing.get("analysis_sha256") != first_provenance["analysis_sha256"]
                    or routing.get("delivery_sha256") != first_provenance["comparison_delivery_sha256"]
                    or routing.get("route") != "PASS" or routing.get("passed_block_metric_checks") != 24
                    or first["route"] != "PASS"):
                raise ValueError("MATRIX_ROUTING_REFERENCE_BINDING_CHANGED")
        all_rows.extend(rows[key] for key in sorted(rows)); inputs.append(provenance)
    by_block, summary = [], []
    for budget in (20, 80):
        for metric in METRICS:
            block_values = []
            for block in BLOCKS:
                selected = [loaded[budget, seed][1][block, metric] for seed in (41, 42, 43)]
                values = {key: _mean([row[key] for row in selected])
                          for key in ("reference", "candidate", "candidate_minus_reference")}
                row = dict(network_budget_epochs=budget, block=block, horizon_ms=500, metric=metric,
                    **values, training_seeds=3, independent_blocks=1,
                    all_seed_checks_passed=all(item["passed"] for item in selected),
                    failed_seed_checks=sum(not item["passed"] for item in selected))
                by_block.append(row); block_values.append(row)
            summary.append(dict(network_budget_epochs=budget, horizon_ms=500, metric=metric,
                **{key: _mean([row[key] for row in block_values])
                   for key in ("reference", "candidate", "candidate_minus_reference")},
                training_seeds=3, independent_blocks=4,
                minimum_block_mean_difference=min(row["candidate_minus_reference"] for row in block_values),
                maximum_block_mean_difference=max(row["candidate_minus_reference"] for row in block_values),
                failed_seed_block_checks=sum(row["failed_seed_checks"] for row in block_values)))
    tables = {"conditional_matrix_by_seed_block.csv": all_rows, "conditional_matrix_by_block.csv": by_block,
              "conditional_matrix_summary.csv": summary}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    result = dict(status="DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_MATRIX", completed_cells=6,
        network_budget_epochs=[20, 80], training_seed_ids=[41, 42, 43], training_seeds=3,
        independent_blocks=4, connectivity_instances=1, reference_arm=REFERENCE, candidate_arm=CANDIDATE,
        horizon_ms=500, metrics=list(METRICS), decisions_per_cell=24, retained_seed_block_metric_decisions=len(all_rows),
        route="PASS" if all(row["passed"] for row in all_rows) else "STOP",
        failed_seed_block_metric_checks=sum(not row["passed"] for row in all_rows), inputs=inputs,
        scientific_protocol=science, aggregation="equal mean of three training seeds within each original block, then equal mean over four blocks",
        uncertainty="Descriptive only. The three seeds share four development blocks;12 seed-block combinations are not independent replicates. No CI, p or q values.",
        scientific_claim_qualified=False, formal_test_accessed=False, calibration_outcomes_accessed=False,
        cross_node_runtime_comparison_qualified=False, matched_end_to_end_speedup_qualified=False,
        new_forecast_origins=0, new_posterior_draws=0, new_source_simulations=0, production_episodes=0,
        implementation_sha256=sha256_file(Path(__file__)),
        artifacts={name: sha256_file(output / name) for name in tables})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "mean", "checkpoint", "field-view"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-readout", type=Path, help="Reuse an existing bound reference readout when available")
    parser.add_argument("--matrix-cell", nargs=3, action="append", metavar=("BUDGET", "SEED", "ARCHIVE"))
    parser.add_argument("--matrix-reference", nargs=3, action="append", metavar=("BUDGET", "SEED", "DELIVERY"))
    args = parser.parse_args()
    if args.matrix_cell:
        if any(getattr(args, name) is not None for name in ("reference", "candidate", "mean", "checkpoint", "field_view", "reference_readout")):
            parser.error("matrix and individual-cell inputs cannot be combined")
        cells = [(int(budget), int(seed), Path(path)) for budget, seed, path in args.matrix_cell]
        references = {(int(budget), int(seed)): Path(path) for budget, seed, path in (args.matrix_reference or [])}
        if len(references) != len(args.matrix_reference or []):
            parser.error("duplicate matrix reference cell")
        result = aggregate_matrix(cells, args.output, references)
    else:
        if args.matrix_reference or any(getattr(args, name) is None for name in ("reference", "candidate", "mean", "checkpoint", "field_view")):
            parser.error("individual comparison requires reference, candidate, mean, checkpoint and field-view")
        result = compare(args.reference, args.candidate, args.reference_readout, args.mean,
                         args.checkpoint, args.field_view, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Descriptive three-seed PFC analysis from authenticated delivered metadata.

This consumer does not load checkpoints, source arrays, or forecast arrays.
Original complete replay receipts establish numerical output validity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .braid_population_comparison import DeliveredCell, _training_cost
from .pfc_parameter_block_execution import CONDITIONS
from .pfc_population_evaluation import validate_forecast_costs
from .pfc_renewal_data import MINIMUM_INTERVAL_POLICY
from .pfc_scale_analysis import SOURCE_ENTRY_SHA256, SOURCE_REVISION
from .population_forecast_analysis import COST_METRICS, _measured_origin_columns
from .provenance import sha256_file
from .trajectory_forecast_analysis import _write_rows

SEEDS = (41, 42, 43)
ORIGINS = (2000, 8000, 48000, 80000)
HORIZONS = (4, 16, 100)
METRICS = ("field_energy_score_hz", "unit_bin_mean_mse_corrected",
           "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
           "unit_total_coverage_90", "unit_total_interval_width_90",
           "refractory_count_violation_1bin", "refractory_count_violation_2bin")
CONTROL_CONTRACT = dict(
    control_timing="observed_source_operation_segments",
    control_window_reduction="time mean of source assignment amplitudes and source schedule indicators",
    admissible_actions="registered cue/distractor source commands; per-unit currents are verification data")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, nonnegative=False):
    require(not isinstance(value, bool), "PFC_ANALYSIS_NUMERIC_REQUIRED:" + name)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("PFC_ANALYSIS_NUMERIC_REQUIRED:" + name) from error
    require(np.isfinite(result) and (not nonnegative or result >= 0),
            "PFC_ANALYSIS_FINITE_REQUIRED:" + name)
    return result


def read(path):
    return json.loads(path.read_text())


def unique_index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), "PFC_ANALYSIS_DUPLICATE_MANIFEST_PATH")
    for name in result:
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts,
                "PFC_ANALYSIS_SAFE_MANIFEST_PATH_REQUIRED")
    return result


class FrozenInputs:
    """Read only metadata explicitly indexed by the frozen preparation."""

    def __init__(self, preparation):
        self.root = Path(preparation)
        self.proof = read(self.root / "preparation.json")
        proof = self.proof
        require(proof.get("schema") == "pfc_population_campaign_preparation_v1"
                and proof.get("status") == "PREPARED_COMPLETE_DEVELOPMENT_COHORT"
                and proof.get("training_seeds") == list(SEEDS)
                and (proof.get("fitting_episodes"), proof.get("validation_episodes")) == (80, 30)
                and proof.get("core_neurons") == 100000 and proof.get("source_dt_ms") == .05
                and proof.get("calibration_or_test_accessed") is False,
                "PFC_ANALYSIS_COMPLETE_FROZEN_PREPARATION_REQUIRED")
        self.files = unique_index(proof["files"], "path")
        self.inputs = unique_index(proof["input_files"], "path")
        self.checked = {}
        self.spec = self.input("inputs/cohort.json")
        require(self.inputs["inputs/cohort.json"]["sha256"] == proof["input_sha256"]
                and set(self.spec) == {"schema", "source_entry", "episodes"}
                and self.spec["schema"] == "pfc_population_development_inputs_v1"
                and len(self.spec["episodes"]) == 110,
                "PFC_ANALYSIS_COMPLETE_ORIGINAL_INPUTS_REQUIRED")
        self.config = self.software("Libs/configs/"+proof.get("training_config","neural_population_pfc_task_training.yaml"), yaml_format=True)
        self.forecast = self.software("Libs/configs/"+proof.get("forecast_config","neural_population_pfc_task_forecast.yaml"), yaml_format=True)
        require(self.config["training_seeds"] == list(SEEDS)
                and self.config["bin_ticks"] == 100
                and self.config["input_family"] == "pfc_source_reproduction"
                and self.forecast["origins_ticks"] == list(ORIGINS)
                and self.forecast["horizons_bins"] == list(HORIZONS)
                and self.forecast["joint_samples"] == 16
                and self.forecast["support_windows"] == [1, 2]
                and self.forecast["formal_evaluation"] is False,
                "PFC_ANALYSIS_FROZEN_CONFIGURATION_REQUIRED")
        self.records, self.validation, self.contracts, self.source_costs = [], [], {}, []
        seen, coverage = set(), Counter()
        for index, row in enumerate(self.spec["episodes"]):
            require(set(row) == {"dataset", "source_run", "episode_id", "partition"}
                    and row["partition"] in ("development_fit", "development_validation"),
                    "PFC_ANALYSIS_EXPLICIT_DEVELOPMENT_INPUT_REQUIRED")
            catalog_name = f"inputs/datasets/{index:03d}/dataset.json"
            source_name = f"inputs/sources/{index:03d}"
            catalog = self.input(catalog_name)
            source = self.input(source_name + "/delivery.json")
            report = self.input(source_name + "/source_run/report.json")
            source_config = catalog["binding"]["resolved_source_config"]
            binding = catalog["binding"]
            require(binding.get("scope") == "source_reproduction"
                    and binding.get("scenario_id") == "pfc_working_memory"
                    and binding.get("source_entry_sha256") == SOURCE_ENTRY_SHA256
                    and binding.get("source_revision") == SOURCE_REVISION
                    and all(binding.get(key) == value for key, value in CONTROL_CONTRACT.items()),
                    "PFC_ANALYSIS_SOURCE_CONTROL_CONTRACT_CHANGED")
            require(catalog["complete_episodes"] == 1 and len(catalog["circuits"]) == 1
                    and len(catalog["shards"]) == 1 and len(catalog["shards"][0]["episodes"]) == 1,
                    "PFC_ANALYSIS_COMPLETE_EPISODE_CATALOG_REQUIRED")
            episode = catalog["shards"][0]["episodes"][0]
            circuit = episode["circuit_id"]
            circuit_record = catalog["circuits"][circuit]
            role = "train" if row["partition"] == "development_fit" else "validation"
            candidate = source_config["benchmark_episode_candidate"]
            identity = candidate["identity"]
            circuit_candidate = source_config["benchmark_circuit_candidate"]
            condition = candidate["condition_id"]
            expected_episode = (f"pfc_working_memory_circuit_{identity['circuit_index']:03d}"
                                f"_block_{identity['block_index']:03d}_condition_{identity['condition_index']:02d}")
            require(row["episode_id"] == episode["episode_id"] == identity["episode_id"] == expected_episode
                    and identity["split"] == candidate["intended_split"] == role
                    and candidate["production_eligible"] is False
                    and condition == CONDITIONS[identity["condition_index"]]
                    and identity["scenario_id"] == "pfc_working_memory"
                    and identity["sample_unit"] == "complete_mechanistic_episode"
                    and identity["circuit_index"] in ((23, 63, 52) if role == "train" else (74,))
                    and identity["block_index"] in ((1, 2) if identity["circuit_index"] == 52 else (0, 1, 2)),
                    "PFC_ANALYSIS_PROSPECTIVE_CONDITION_IDENTITY_CHANGED")
            require(circuit in self.config["fit_circuit_ids" if role == "train" else "validation_circuit_ids"]
                    and episode["split"] == "development" and episode["duration_ticks"] == 100000
                    and episode["dt_ms"] == .05 and circuit_record["units"] == 100000
                    and source_config["excitatory_neurons"] == 80000
                    and source_config["inhibitory_neurons"] == 20000
                    and source_config["excitatory_preference_bins"] == 16,
                    "PFC_ANALYSIS_SOURCE_MAP_OR_CLOCK_CHANGED")
            require(circuit_candidate["circuit_index"] == identity["circuit_index"]
                    and circuit_candidate["intended_split"] == role
                    and circuit_candidate["production_eligible"] is False
                    and circuit == "pfc_" + circuit_candidate["parameter_identity"][:16]
                    and circuit_record["provenance"]["parameter_identity"] == circuit_candidate["parameter_identity"]
                    and circuit_record["provenance"]["connectivity_identity"] == circuit_candidate["connectivity_identity"],
                    "PFC_ANALYSIS_CIRCUIT_IDENTITY_CHANGED")
            require(str(Path(row["dataset"]).resolve()) == str(Path(self.inputs[catalog_name]["source"]).parent.resolve())
                    and str(Path(row["source_run"]).resolve()) ==
                    str(Path(self.inputs[source_name + "/source_run/report.json"]["source"]).parent.resolve()),
                    "PFC_ANALYSIS_ORIGINAL_INPUT_PATH_CHANGED")
            reference = binding["source_bindings"]
            require(len(reference) == 1 and reference[0]["case_id"] == row["episode_id"]
                    and Path(reference[0]["path"]).resolve() == Path(row["source_run"]).resolve()
                    and reference[0]["delivery_sha256"] == self.inputs[source_name + "/delivery.json"]["sha256"]
                    and reference[0]["report_sha256"] == self.inputs[source_name + "/source_run/report.json"]["sha256"],
                    "PFC_ANALYSIS_SOURCE_RECEIPT_BINDING_CHANGED")
            native_block = f"pfc_initial_{identity['initial_state_seed']}_external_{identity['background_seed']}"
            require(episode["block_id"] == native_block
                    and source.get("status") == "ARCHIVED_PFC_SOURCE_EXECUTION"
                    and source.get("source_execution_complete") is True and source.get("exit_code") == 0
                    and source["case_id"] == report["case_id"] == row["episode_id"]
                    and report.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE"
                    and report.get("source_execution_complete") is True
                    and report["resolved_config"] == source_config
                    and report["config_sha256"] == binding["config_sha256"]
                    and report["seeds"] == dict(initial_rng_seed=identity["initial_state_seed"],
                                               external_noise_seed=identity["background_seed"])
                    and report["random_stream_policy"] == "independent_initial_external",
                    "PFC_ANALYSIS_NATIVE_SOURCE_IDENTITY_CHANGED")
            source_files = unique_index(source["files"], "path")
            require(source_files["source_run/report.json"]["sha256"] == reference[0]["report_sha256"]
                    and self.inputs[source_name + "/source_run/config.yaml"]["sha256"] == binding["config_sha256"],
                    "PFC_ANALYSIS_SOURCE_REPORT_CHANGED")
            for relative, metadata in [(circuit_record["path"], circuit_record),
                                       (catalog["shards"][0]["path"], catalog["shards"][0])]:
                require(self.inputs[f"inputs/datasets/{index:03d}/" + relative]["sha256"] == metadata["sha256"],
                        "PFC_ANALYSIS_DATASET_PAYLOAD_MANIFEST_CHANGED")
            source_binding = dict(dataset_sha256=self.inputs[catalog_name]["sha256"], episode_id=row["episode_id"],
                circuit_id=circuit, block_id=native_block, split="development", source_revision=SOURCE_REVISION,
                source_entry_sha256=SOURCE_ENTRY_SHA256, source_report_sha256=reference[0]["report_sha256"],
                source_delivery_sha256=reference[0]["delivery_sha256"], minimum_interval_policy=MINIMUM_INTERVAL_POLICY,
                initial_availability="source_all_ready; no unobserved prior spikes", bin_ticks=100, source_dt_ms=.05)
            self.records.append(dict(partition=row["partition"], **source_binding))
            require(row["episode_id"] not in seen, "PFC_ANALYSIS_DUPLICATE_COMPLETE_EPISODE")
            seen.add(row["episode_id"])
            coverage[(identity["circuit_index"], identity["block_index"], condition)] += 1
            if role == "validation":
                case = dict(case_id=circuit + "__" + row["episode_id"], circuit_id=circuit,
                    episode_id=row["episode_id"], circuit_index=identity["circuit_index"],
                    block_index=identity["block_index"], condition_id=condition, block=native_block,
                    source_binding=source_binding)
                self.validation.append(case)
                self.contracts[case["case_id"]] = dict(block=native_block, unit_count=100000, field_count=17,
                    bin_ms=5., source_dt_ms=.05, episode_bins=1000, full_unit_retention="all_origins",
                    episode_inputs_materialized=False)
                cost = report["simulator_cost"]
                require(isinstance(cost.get("scope"), str) and cost["scope"], "PFC_ANALYSIS_SOURCE_COST_SCOPE_REQUIRED")
                self.source_costs.append(dict(case_id=case["case_id"], circuit_index=74,
                    block_index=case["block_index"], condition_id=condition, source_block=native_block,
                    source_report_sha256=reference[0]["report_sha256"],
                    simulator_wall_seconds=number(cost["wall_seconds"], "source wall", nonnegative=True),
                    simulator_cost_scope=cost["scope"], simulator_cost=json.dumps(cost, sort_keys=True)))
        expected = {(circuit, block, condition) for circuit in (23, 63, 74, 52)
                    for block in ((1, 2) if circuit == 52 else (0, 1, 2)) for condition in CONDITIONS}
        require(set(coverage) == expected and all(count == 1 for count in coverage.values())
                and len(self.validation) == 30, "PFC_ANALYSIS_COMPLETE_CONDITION_ENVELOPE_REQUIRED")

    def _metadata(self, path, row, *, yaml_format=False):
        require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
                "PFC_ANALYSIS_FROZEN_METADATA_CHANGED:" + str(path))
        self.checked[str(path)] = row["sha256"]
        return yaml.safe_load(path.read_text()) if yaml_format else read(path)

    def input(self, name):
        require(name in self.inputs, "PFC_ANALYSIS_MISSING_FROZEN_INPUT:" + name)
        row = self.inputs[name]
        return self._metadata(Path(row["source"]), row)

    def software(self, name, *, yaml_format=False):
        require(name in self.files, "PFC_ANALYSIS_MISSING_FROZEN_SOFTWARE:" + name)
        return self._metadata(self.root / name, self.files[name], yaml_format=yaml_format)


def score_rows(path, seed, frozen):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "PFC_ANALYSIS_UNIQUE_SCORE_COLUMNS_REQUIRED")
        raw = list(reader)
    cases = {case["case_id"]: case for case in frozen.validation}
    expected = {(case, origin, horizon) for case in cases for origin in ORIGINS for horizon in HORIZONS}
    result, seen = [], set()
    for row in raw:
        key = (row["case_id"], int(row["origin_tick"]), int(row["horizon_bins"]))
        require(key in expected and key not in seen and int(row["training_seed"]) == seed
                and row["circuit_id"] == cases[key[0]]["circuit_id"], "PFC_ANALYSIS_SCORE_IDENTITY_CHANGED")
        seen.add(key)
        case = cases[key[0]]
        values = {}
        for metric in METRICS:
            value = row[metric]
            width = int(metric.split("_")[-1].removesuffix("bin")) if metric.startswith("refractory_") else None
            if value == "" or value is None:
                require(width is not None and width > key[2], "PFC_ANALYSIS_UNEXPECTED_UNDEFINED_METRIC:" + metric)
                values[metric] = None
            else:
                values[metric] = number(value, metric, nonnegative=metric != "unit_bin_mean_mse_corrected")
                if metric == "unit_total_coverage_90" or width is not None:
                    require(values[metric] <= 1, "PFC_ANALYSIS_PROBABILITY_METRIC_RANGE:" + metric)
        costs = {name: None if name == "peak_gpu_allocated_bytes" and row.get(name) in ("", None)
                 else number(row[name], name, nonnegative=True) for name in COST_METRICS}
        result.append(dict(seed=seed, training_seed=seed, case_id=key[0], circuit_id=case["circuit_id"],
            circuit_index=74, block_index=case["block_index"], source_block=case["block"],
            condition_id=case["condition_id"], origin_tick=key[1], horizon_bins=key[2],
            origin_ms=key[1] * .05, horizon_ms=key[2] * 5., **values, **costs))
    require(seen == expected and len(raw) == 360, "PFC_ANALYSIS_COMPLETE_SCORE_GRID_REQUIRED")
    return result


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        row = dict(zip(keys, group), descriptive_rows=len(values))
        for metric in METRICS:
            present = [value[metric] for value in values if value[metric] is not None]
            row.update({metric + "_defined": len(present)})
            row.update({metric + "_" + name: (float(function(present)) if present else None)
                        for name, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max))})
        result.append(row)
    return result


def analyze(campaign, preparation, output, *, seed_cells=None):
    campaign, output = Path(campaign), Path(output)
    require(not output.exists(), "PFC_ANALYSIS_NEW_OUTPUT_REQUIRED")
    overrides = seed_cells or {}
    require(set(overrides) <= set(SEEDS), "PFC_ANALYSIS_REGISTERED_SEED_OVERRIDE_REQUIRED")
    frozen = FrozenInputs(preparation)
    rows, costs, training, receipts, origin_identity = [], [], [], {}, None
    for seed in SEEDS:
        cell = DeliveredCell(Path(overrides.get(seed, campaign / f"seed_{seed}")), seed)
        require(cell.proof.get("input_family") == "pfc_source_reproduction"
                and cell.proof.get("calibration_or_test_accessed") is False
                and (cell.proof.get("fit_episodes"), cell.proof.get("validation_episodes")) == (80, 30)
                and cell.proof.get("preparation_sha256") == sha256_file(frozen.root / "preparation.json"),
                "PFC_ANALYSIS_MATCHED_PFC_DELIVERY_REQUIRED")
        run, evaluation, replay = (cell.read(name) for name in
                                  ("training/run.json", "forecasts/evaluation.json", "replay.json"))
        require(run.get("status") == "FITTED_DEVELOPMENT" and run.get("training_seed") == seed
                and run.get("input_family") == "pfc_source_reproduction"
                and (run.get("fitting_episodes"), run.get("validation_episodes")) == (80, 30)
                and run["binding"]["config"] == dict(frozen.config, device=cell.proof["device"],
                                                     cpu_threads=2, representation=cell.proof["representation"])
                and run["binding"]["seed"] == seed
                and run["binding"]["circuit_inputs_sha256"] == frozen.proof["input_sha256"]
                and run["binding"]["complete_episode_bindings"] == frozen.records
                and run.get("calibration_outcomes_accessed") is False and run.get("test_outcomes_accessed") is False
                and run.get("scientific_claim_qualified") is False and run.get("stop_reason"),
                "PFC_ANALYSIS_TRAINING_SOURCE_OR_CONFIG_CHANGED")
        maximum = frozen.config["maximum_epochs"]
        completed, best = run["completed_epochs"], run["best_epoch"]
        require(type(completed) is int and type(best) is int and 0 <= best < completed <= maximum
                and run.get("effective_maximum_epochs") == maximum,
                "PFC_ANALYSIS_REGISTERED_TRAINING_BUDGET_REQUIRED")
        stale = completed - best - 1
        patience = frozen.config["early_stopping_patience"]
        require((run["stop_reason"] == "epoch_budget" and completed == maximum)
                or (run["stop_reason"] == "validation_patience" and stale >= patience),
                "PFC_ANALYSIS_REGISTERED_TRAINING_STOP_REQUIRED")
        for name, digest in run["trainer_sources"].items():
            path = "Libs/configs/" + name if name.endswith(".yaml") else "Libs/population_dynamics/" + name
            require(path in frozen.files and frozen.files[path]["sha256"] == digest,
                    "PFC_ANALYSIS_TRAINER_SNAPSHOT_CHANGED")
        require("training/best.pt" in cell.files
                and cell.files["training/best.pt"]["sha256"] == run["best_checkpoint_sha256"],
                "PFC_ANALYSIS_CHECKPOINT_MANIFEST_CHANGED")
        require(evaluation.get("status") == "EVALUATED_PFC_DEVELOPMENT_CHECKPOINT"
                and evaluation.get("training_seed") == seed and evaluation.get("configuration") == frozen.forecast
                and evaluation.get("checkpoint_sha256") == run["best_checkpoint_sha256"]
                and evaluation.get("original_training_inputs_sha256") == frozen.proof["input_sha256"]
                and evaluation.get("inputs_sha256") == frozen.proof["input_sha256"]
                and evaluation.get("scientific_claim_qualified") is False
                and evaluation.get("formal_evaluation") is False and evaluation.get("speedup_qualified") is False
                and evaluation.get("episodes") == 30 and evaluation.get("score_rows") == 360,
                "PFC_ANALYSIS_EVALUATION_BINDING_CHANGED")
        evaluator_name = "Libs/population_dynamics/pfc_population_evaluation.py"
        require(evaluation["evaluator_sha256"] == frozen.files[evaluator_name]["sha256"],
                "PFC_ANALYSIS_EVALUATOR_SNAPSHOT_CHANGED")
        expected = [(case["case_id"], origin) for case in frozen.validation for origin in ORIGINS]
        require([(row["case_id"], row["origin_tick"]) for row in evaluation["origins"]] == expected
                and [(row["case_id"], row["origin_tick"]) for row in replay["origins"]] == expected
                and replay.get("status") == "PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY"
                and replay.get("checkpoint_sha256") == run["best_checkpoint_sha256"]
                and replay.get("scientific_claim_qualified") is False
                and replay.get("unit_values_consumed") == 16 * 100 * 100000 * 120
                and all(row["unit_values"] == 16 * 100 * 100000 for row in replay["origins"]),
                "PFC_ANALYSIS_COMPLETE_REPLAY_REQUIRED")
        identity_rows = []
        for index, origin in enumerate(evaluation["origins"]):
            case = frozen.validation[index // len(ORIGINS)]
            require(origin["source_binding"] == case["source_binding"]
                    and origin["units"] == 100000 and origin["fields"] == 17,
                    "PFC_ANALYSIS_ORIGIN_SOURCE_BINDING_CHANGED")
            identity_rows.append((origin["case_id"], origin["origin_tick"],
                                  origin["source_binding"], origin["source_metadata_identity"]))
        require(origin_identity is None or origin_identity == identity_rows,
                "PFC_ANALYSIS_CROSS_SEED_SOURCE_METADATA_CHANGED")
        origin_identity = identity_rows
        expected_artifacts = {"scores.csv"} | {row["artifact"] for row in evaluation["origins"]}
        require(len(expected_artifacts) == 121 and set(evaluation["artifacts"]) == expected_artifacts
                and all(not Path(name).is_absolute() and ".." not in Path(name).parts
                        for name in expected_artifacts),
                "PFC_ANALYSIS_COMPLETE_ARTIFACT_MANIFEST_REQUIRED")
        for name, digest in evaluation["artifacts"].items():
            require("forecasts/" + name in cell.files and cell.files["forecasts/" + name]["sha256"] == digest,
                    "PFC_ANALYSIS_REPLAYED_ARTIFACT_BINDING_CHANGED")
        values = score_rows(cell.path("forecasts/scores.csv"), seed, frozen)
        measured = validate_forecast_costs(evaluation, values, validation=frozen.validation,
                                           case_cost_contracts=frozen.contracts)
        require(measured is not None, "PFC_ANALYSIS_MEASURED_COSTS_REQUIRED")
        for origin in measured["origin_costs"]:
            case = next(case for case in frozen.validation if case["case_id"] == origin["case_id"])
            costs.append(dict(seed=seed, case_id=case["case_id"], circuit_index=74,
                block_index=case["block_index"], condition_id=case["condition_id"],
                source_block=case["block"], origin_ms=origin["origin_ms"],
                **{name: origin[name] for name in COST_METRICS},
                **_measured_origin_columns({"measured_evaluation_costs": measured}, case["case_id"], origin["origin_ms"])))
        rows.extend(values)
        training.append(_training_cost(cell, run, "population_" + run["representation"]))
        receipts[str(seed)] = dict(path=str(cell.root), **cell.binding(),
            checkpoint_sha256=run["best_checkpoint_sha256"],
            evaluation_costs=measured["function_costs"],
            replayed_array_bindings=len(evaluation["origins"]))
    output.mkdir(parents=True)
    artifacts = {}
    tables = (("all_seeds.csv", rows),
              ("descriptive_by_circuit_block_horizon.csv", summarize(rows, ("circuit_index", "block_index", "horizon_ms"))),
              ("descriptive_by_seed_horizon.csv", summarize(rows, ("seed", "horizon_ms"))),
              ("costs_by_origin.csv", costs), ("training_costs.csv", training),
              ("source_episode_costs.csv", frozen.source_costs))
    for name, values in tables:
        _write_rows(output / name, values)
        artifacts[name] = sha256_file(output / name)
    report = dict(status="COMPLETE_DESCRIPTIVE_PFC_POPULATION_FORECAST_ANALYSIS", training_seeds=list(SEEDS),
        score_rows=len(rows), origin_cost_rows=len(costs), source_episode_cost_rows=len(frozen.source_costs),
        validation_circuits=[74], validation_initial_noise_blocks=[0, 1, 2], conditions=list(CONDITIONS),
        consumed_deliveries=receipts, preparation=str(frozen.root),
        preparation_sha256=sha256_file(frozen.root / "preparation.json"),
        original_input_sha256=frozen.proof["input_sha256"], consumed_metadata=frozen.checked, artifacts=artifacts,
        array_consumption_scope="Original delivery and exhaustive replay bindings reused; NPZ/checkpoint bytes not read or rehashed here.",
        source_cost_scope="One complete simulator episode per condition/block, separate from evaluation and training; no amortization or matched-fidelity claim.",
        summary_scope="Mean/std/min/max across recorded rows; training seeds, conditions and origins are not independent circuit replicates.",
        formal_evaluation=False, inferential_statistics_performed=False, confidence_intervals_computed=False,
        fdr_performed=False, baseline_comparison_performed=False, distribution_baseline_comparison_performed=False,
        scientific_claim_qualified=False, speedup_qualified=False, full_end_to_end_cost_matched=False,
        samples_regenerated=False, source_simulation_repeated=False, training_repeated=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "preparation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}-cell", type=Path,
                            help="Explicit successful resource-recovery cell; original archive remains preserved.")
    args = parser.parse_args()
    overrides = {seed: getattr(args, f"seed{seed}_cell") for seed in SEEDS
                 if getattr(args, f"seed{seed}_cell") is not None}
    result = analyze(args.campaign, args.preparation, args.output, seed_cells=overrides)
    print(json.dumps({key: result[key] for key in ("status", "score_rows", "origin_cost_rows")}))


if __name__ == "__main__":
    main()

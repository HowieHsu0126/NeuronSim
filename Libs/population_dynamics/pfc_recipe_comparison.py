"""Compare accepted PFC development recipes without generating new samples."""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import yaml

from .pfc_population_forecast_analysis import (
    METRICS, SEEDS, number, require, summarize, unique_index,
)
from .trajectory_forecast_analysis import _write_rows

IDENTITY = ("seed", "case_id", "circuit_id", "circuit_index", "block_index",
            "source_block", "condition_id", "origin_tick", "horizon_bins")
INTEGER_COLUMNS = ("seed", "training_seed", "circuit_index", "block_index",
                   "origin_tick", "horizon_bins")
SCORING_FUNCTIONS = ("_scores", "_forecast", "validate_forecast_costs", "replay")
SCORING_HELPERS = ("trajectory_sample_metrics.py", "trajectory_scores.py")
RANDOM_FORECAST_KEYS = frozenset(("sampling_seed_base", "random_seed_policy"))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _function(source: str, name: str) -> str:
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
    require(len(functions) == 1, "PFC_RECIPE_FUNCTION_REQUIRED:" + name)
    return ast.dump(functions[0], include_attributes=False)


@dataclass
class SavedAnalysis:
    root: Path
    report: dict
    preparation: Path
    proof: dict
    training: dict
    forecast: dict
    rows: list[dict]
    costs: list[dict]
    consumed: dict[str, str]

    def software(self, name: str) -> str:
        files = unique_index(self.proof["files"], "path")
        key = "Libs/population_dynamics/" + name
        return _checked(self.preparation / key, files[key], self.consumed).decode()


def _checked(path: Path, binding: dict, consumed: dict[str, str]) -> bytes:
    data = path.read_bytes()
    digest = _digest(data)
    require(digest == binding["sha256"]
            and ("bytes" not in binding or len(data) == binding["bytes"]),
            "PFC_RECIPE_BOUND_FILE_CHANGED:" + str(path))
    consumed[str(path)] = digest
    return data


def _load(root: Path) -> SavedAnalysis:
    root = root.resolve()
    delivery_path = root / "delivery.json"
    data = delivery_path.read_bytes()
    delivery = json.loads(data)
    require(delivery["status"] == "PASS", "PFC_RECIPE_COMPLETE_ANALYSIS_REQUIRED")
    consumed = {str(delivery_path): _digest(data)}
    files = unique_index(delivery["files"], "path")

    def artifact(name):
        return _checked(root / name, files[name], consumed).decode()

    report = json.loads(artifact("report.json"))
    require(report["status"] == "COMPLETE_DESCRIPTIVE_PFC_POPULATION_FORECAST_ANALYSIS"
            and report["training_seeds"] == list(SEEDS)
            and report["scientific_claim_qualified"] is False
            and report["formal_evaluation"] is False,
            "PFC_RECIPE_COMPLETE_DEVELOPMENT_ANALYSIS_REQUIRED")
    preparation = Path(report["preparation"])
    proof = json.loads(_checked(preparation / "preparation.json",
                                {"sha256": report["preparation_sha256"]}, consumed))
    require(proof["schema"] == "pfc_population_campaign_preparation_v1"
            and proof["training_seeds"] == list(SEEDS)
            and proof["input_sha256"] == report["original_input_sha256"]
            and proof["calibration_or_test_accessed"] is False,
            "PFC_RECIPE_PREPARATION_BINDING_REQUIRED")
    software = unique_index(proof["files"], "path")

    def config(kind):
        default = "neural_population_pfc_task_" + kind + ".yaml"
        key = "Libs/configs/" + proof.get(kind + "_config", default)
        return yaml.safe_load(_checked(preparation / key, software[key], consumed))

    training, forecast = config("training"), config("forecast")
    rows = list(csv.DictReader(artifact("all_seeds.csv").splitlines()))
    for row in rows:
        for key in INTEGER_COLUMNS:
            value = number(row[key], key)
            require(value.is_integer(), "PFC_RECIPE_INTEGER_ID_REQUIRED:" + key)
            row[key] = int(value)
        require(row["training_seed"] == row["seed"], "PFC_RECIPE_TRAINING_SEED_MISMATCH")
        for key in ("origin_ms", "horizon_ms"):
            row[key] = number(row[key], key, nonnegative=True)
        require(row["origin_ms"] == row["origin_tick"] * proof["source_dt_ms"]
                and row["horizon_ms"] == row["horizon_bins"] * training["bin_ticks"]
                * proof["source_dt_ms"], "PFC_RECIPE_CLOCK_MISMATCH")
        for key in METRICS:
            row[key] = None if row[key] == "" else number(row[key], key)
        require(row["observed_spikes_per_unit"] is not None,
                "PFC_RECIPE_OBSERVATION_REQUIRED")

    index = {tuple(row[key] for key in IDENTITY): row for row in rows}
    require(len(index) == len(rows), "PFC_RECIPE_DUPLICATE_CONDITION")
    cases = {row["case_id"] for row in rows}
    expected = {(seed, case, origin, horizon) for seed in SEEDS for case in cases
                for origin in forecast["origins_ticks"]
                for horizon in forecast["horizons_bins"]}
    actual = {(r["seed"], r["case_id"], r["origin_tick"], r["horizon_bins"]) for r in rows}
    require(len(cases) == proof["validation_episodes"]
            and len(rows) == report["score_rows"] == len(expected)
            and actual == expected, "PFC_RECIPE_INCOMPLETE_CONDITION_GRID")
    require({r["condition_id"] for r in rows} == set(report["conditions"])
            and {r["block_index"] for r in rows} == set(report["validation_initial_noise_blocks"])
            and {r["circuit_index"] for r in rows} == set(report["validation_circuits"]),
            "PFC_RECIPE_SOURCE_METADATA_MISMATCH")
    costs = list(csv.DictReader(artifact("training_costs.csv").splitlines()))
    require(len(costs) == len(SEEDS) and {int(row["seed"]) for row in costs} == set(SEEDS),
            "PFC_RECIPE_COMPLETE_TRAINING_COSTS_REQUIRED")
    return SavedAnalysis(root, report, preparation, proof, training, forecast,
                         rows, costs, consumed)




def _evaluation_function(analysis: SavedAnalysis, name: str) -> str:
    """Compare delivered numerical code across inline and shared source layouts.

    Replay admission is source-specific. Its numerical signature retains the
    primitive imports and every CSV/array, denominator, metric and cost check.
    The new replay also authenticates all shared implementation fingerprints.
    """
    source = analysis.software("pfc_population_evaluation.py")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    forwarded = False
    if len(functions) == 1:
        body = [node for node in functions[0].body if not (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        forwarded = (len(body) == 1 and isinstance(body[0], ast.Return)
                     and isinstance(body[0].value, ast.Call) and isinstance(body[0].value.func, ast.Attribute)
                     and isinstance(body[0].value.func.value, ast.Name)
                     and body[0].value.func.value.id == "shared" and body[0].value.func.attr == name)
    direct_import = any(isinstance(node, ast.ImportFrom) and node.level == 1
                        and node.module == "source_population_evaluation"
                        and any(item.name == name and item.asname is None for item in node.names)
                        for node in tree.body)
    module_import = any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
                        and any(item.name == "source_population_evaluation" and item.asname == "shared"
                                for item in node.names) for node in tree.body)
    if not functions or forwarded:
        require((forwarded and module_import) or (not functions and direct_import),
                "PFC_RECIPE_EVALUATOR_LAYOUT_UNSUPPORTED:" + name)
        source = analysis.software("source_population_evaluation.py")
    if name == "replay":
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(functions) == 1, "PFC_RECIPE_FUNCTION_REQUIRED:" + name)
        function = functions[0]
        indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.With)
                   and any(isinstance(item, ast.Constant) and item.value == "scores.csv" for item in ast.walk(node))
                   and any(isinstance(item, ast.Name) and item.id == "forecasts" for item in ast.walk(node))]
        if len(indices) == 1:
            imports = [node for node in function.body[:indices[0]] if isinstance(node, (ast.Import, ast.ImportFrom))]
            return ast.dump(ast.Module(body=imports + function.body[indices[0]:], type_ignores=[]), include_attributes=False)
    return _function(source, name)


def _initialization_function(analysis: SavedAnalysis) -> str:
    """Resolve the initializer from either supported delivered source layout."""
    source = analysis.software("pfc_population_training_data.py")
    tree = ast.parse(source)
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name == "initialize_observation" for node in ast.walk(tree)):
        return _function(source, "initialize_observation")
    imported = any(isinstance(node, ast.ImportFrom) and node.level == 1
                   and node.module == "source_population_training"
                   and any(name.name == "SourcePopulationTrainingCohort" and name.asname is None
                           for name in node.names) for node in tree.body)
    inherited = any(isinstance(node, ast.ClassDef) and node.name == "PfcPopulationTrainingCohort"
                    and any(isinstance(base, ast.Name) and base.id == "SourcePopulationTrainingCohort"
                            for base in node.bases) for node in tree.body)
    require(imported and inherited, "PFC_RECIPE_INITIALIZER_LAYOUT_UNSUPPORTED")
    return _function(analysis.software("source_population_training.py"), "initialize_observation")


def _differences(reference: dict, candidate: dict) -> dict:
    return {key: {"reference": reference.get(key), "candidate": candidate.get(key)}
            for key in sorted(set(reference) | set(candidate))
            if reference.get(key) != candidate.get(key)}


def _summary(reference: list[dict], candidate: list[dict], delta: list[dict],
             keys: tuple[str, ...]) -> list[dict]:
    result = {}
    for label, rows in (("reference", reference), ("candidate", candidate), ("delta", delta)):
        for row in summarize(rows, keys):
            group = tuple(row[key] for key in keys)
            merged = result.setdefault(group, {key: row[key] for key in keys})
            merged["paired_rows"] = row["descriptive_rows"]
            merged.update({label + "_" + key: value for key, value in row.items()
                           if key not in keys and key != "descriptive_rows"})
    return [result[key] for key in sorted(result)]


def compare(reference: Path, candidate: Path, output: Path) -> dict:
    """Keep all paired conditions and report candidate-minus-reference differences."""
    require(not output.exists(), "PFC_RECIPE_OUTPUT_EXISTS_REUSE_OR_CHOOSE_NEW_PATH")
    ref, cand = _load(reference), _load(candidate)
    input_ref = unique_index(ref.proof["input_files"], "path")
    input_cand = unique_index(cand.proof["input_files"], "path")
    bindings = lambda values: {key: (row["sha256"], row["bytes"])
                               for key, row in values.items()}
    require(bindings(input_ref) == bindings(input_cand)
            and ref.report["original_input_sha256"] == cand.report["original_input_sha256"],
            "PFC_RECIPE_SOURCE_INPUTS_DIFFER")
    ref_protocol = {k: v for k, v in ref.forecast.items() if k not in RANDOM_FORECAST_KEYS}
    cand_protocol = {k: v for k, v in cand.forecast.items() if k not in RANDOM_FORECAST_KEYS}
    require(ref_protocol == cand_protocol, "PFC_RECIPE_SCORING_PROTOCOL_DIFFERS")
    for name in SCORING_FUNCTIONS:
        require(_evaluation_function(ref, name) == _evaluation_function(cand, name),
                "PFC_RECIPE_SCORING_FUNCTION_DIFFERS:" + name)
    for name in SCORING_HELPERS:
        require(ref.software(name) == cand.software(name),
                "PFC_RECIPE_SCORING_HELPER_DIFFERS:" + name)
    initializer_equal = _initialization_function(ref) == _initialization_function(cand)
    left = {tuple(row[key] for key in IDENTITY): row for row in ref.rows}
    right = {tuple(row[key] for key in IDENTITY): row for row in cand.rows}
    require(left.keys() == right.keys(), "PFC_RECIPE_PAIRED_IDENTITIES_DIFFER")
    paired, deltas = [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        require(a["observed_spikes_per_unit"] == b["observed_spikes_per_unit"],
                "PFC_RECIPE_OBSERVATIONS_DIFFER")
        identity = {name: a[name] for name in (*IDENTITY, "origin_ms", "horizon_ms")}
        row, delta = dict(identity), dict(identity)
        for metric in METRICS:
            row["reference_" + metric], row["candidate_" + metric] = a[metric], b[metric]
            difference = None if a[metric] is None or b[metric] is None else b[metric] - a[metric]
            row["delta_" + metric] = difference
            delta[metric] = difference
        paired.append(row)
        deltas.append(delta)
    training_ref, training_cand = dict(ref.training), dict(cand.training)
    for config in (training_ref, training_cand):
        config.setdefault("control_geometry", "static")
    shared = set(row["path"] for row in ref.proof["files"]) & set(row["path"] for row in cand.proof["files"])
    source_ref, source_cand = (unique_index(item.proof["files"], "path") for item in (ref, cand))
    changed_sources = {name: {"reference": source_ref[name]["sha256"], "candidate": source_cand[name]["sha256"]}
                       for name in sorted(shared) if name.endswith(".py")
                       and source_ref[name]["sha256"] != source_cand[name]["sha256"]}
    outputs = {
        "paired_metrics.csv": paired,
        "by_horizon.csv": _summary(ref.rows, cand.rows, deltas, ("horizon_ms",)),
        "by_seed_horizon.csv": _summary(ref.rows, cand.rows, deltas, ("seed", "horizon_ms")),
        "by_source_condition_origin_horizon.csv": _summary(
            ref.rows, cand.rows, deltas,
            ("case_id", "circuit_id", "block_index", "source_block", "condition_id",
             "origin_tick", "origin_ms", "horizon_ms")),
        "training_recipes.csv": [dict(recipe=label, **row) for label, item in
                                 (("reference", ref), ("candidate", cand)) for row in item.costs],
    }
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in outputs.items():
        path = output / name
        _write_rows(path, rows)
        data = path.read_bytes()
        artifacts[name] = {"sha256": _digest(data), "bytes": len(data), "rows": len(rows)}
    code = Path(__file__)
    summary_code = Path(__file__).with_name("pfc_population_forecast_analysis.py")
    result = dict(
        status="COMPLETE_PAIRED_PFC_DEVELOPMENT_RECIPE_COMPARISON",
        reference=str(ref.root), candidate=str(cand.root),
        difference_definition="candidate minus reference; negative error-score deltas favor candidate",
        training_seeds=list(SEEDS), paired_rows=len(paired),
        source_input_file_bindings=len(input_ref), source_episodes=ref.proof["validation_episodes"],
        validation_circuits=ref.report["validation_circuits"],
        validation_initial_noise_blocks=ref.report["validation_initial_noise_blocks"],
        scoring_functions_equal=list(SCORING_FUNCTIONS), scoring_helpers_equal=list(SCORING_HELPERS),
        forecast_configuration_differences=_differences(ref.forecast, cand.forecast),
        training_configuration_differences=_differences(training_ref, training_cand),
        initialization_function="PfcPopulationTrainingCohort.initialize_observation",
        rate_initialization_function_equal=initializer_equal, changed_software=changed_sources,
        matched_forecast_random_streams=all(ref.forecast.get(k) == cand.forecast.get(k)
                                          for k in RANDOM_FORECAST_KEYS),
        consumed_inputs={"reference": ref.consumed, "candidate": cand.consumed},
        implementation={str(code): _digest(code.read_bytes()),
                        str(summary_code): _digest(summary_code.read_bytes())},
        artifacts=artifacts,
        summary_scope="Descriptive mean/std/min/max; rows, training seeds and conditions are not independent circuit replicates.",
        recorded_cost_scope="Delivered attempt costs and hardware retained; no matched-fidelity or matched-hardware speedup.",
        control_geometry_causal_effect_identified=False,
        scientific_claim_qualified=False, inferential_statistics_performed=False,
        confidence_intervals_computed=False, fdr_performed=False,
        samples_regenerated=False, training_repeated=False, source_simulation_repeated=False,
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.output)
    print(json.dumps({key: report[key] for key in ("status", "paired_rows")}))


if __name__ == "__main__":
    main()

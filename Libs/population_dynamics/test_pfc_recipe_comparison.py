"""Saved PFC comparison rejects mismatched evidence without generating samples."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from . import pfc_recipe_comparison as comparison


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def binding(path, root):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(root)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(root):
    proof_path = root / "prepared/preparation.json"
    proof = json.loads(proof_path.read_text())
    proof["files"] = [binding(path, proof_path.parent) for path in
                      sorted((proof_path.parent / "Libs").rglob("*")) if path.is_file()]
    write(proof_path, proof)
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    report["preparation_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    write(report_path, report)
    write(root / "delivery.json", dict(status="PASS", files=[
        binding(root / name, root) for name in ("report.json", "all_seeds.csv", "training_costs.csv")]))


def make(root, candidate):
    root.mkdir()
    prepared = root / "prepared"
    source = prepared / "Libs/population_dynamics"
    source.mkdir(parents=True)
    evaluator = "\n".join("def " + name + "():\n    return 1\n" for name in comparison.SCORING_FUNCTIONS)
    (source / "pfc_population_evaluation.py").write_text(evaluator)
    for name in comparison.SCORING_HELPERS:
        (source / name).write_text("metric_definition = 1\n")
    initialization = "self._intrinsic_unit_features()" if candidate else "self.unit_features"
    (source / "pfc_population_training_data.py").write_text(
        "class Cohort:\n    def initialize_observation(self):\n        return " + initialization + "\n")
    configs = prepared / "Libs/configs"
    configs.mkdir()
    training = dict(bin_ticks=100, maximum_epochs=20, early_stopping_patience=4,
                    training_variational_seed_base=41 if candidate else 61001)
    if candidate:
        training["control_geometry"] = "actuation_only"
    forecast = dict(origins_ticks=[2000, 8000, 48000, 80000], horizons_bins=[4, 16, 100],
                    joint_samples=16, field_units="source_field_mean_unit_rate_hz",
                    sampling_seed_base=41 if candidate else 79001)
    for name, config in (("training", training), ("forecast", forecast)):
        (configs / ("neural_population_pfc_task_" + name + ".yaml")).write_text(yaml.safe_dump(config))
    write(prepared / "preparation.json", dict(
        schema="pfc_population_campaign_preparation_v1", training_seeds=[41, 42, 43],
        input_sha256="a" * 64, calibration_or_test_accessed=False, validation_episodes=30,
        source_dt_ms=.05, files=[], input_files=[dict(path="inputs/source", bytes=100, sha256="b" * 64)]))
    rows = []
    for seed in (41, 42, 43):
        for block in range(3):
            for condition in range(10):
                for origin in forecast["origins_ticks"]:
                    for horizon in forecast["horizons_bins"]:
                        row = dict(seed=seed, training_seed=seed, case_id=f"case{block}_{condition}",
                                   circuit_id="circuit", circuit_index=74, block_index=block,
                                   source_block=f"source{block}", condition_id=f"condition{condition}",
                                   origin_tick=origin, horizon_bins=horizon,
                                   origin_ms=origin * .05, horizon_ms=horizon * 5.)
                        row.update({metric: 1. for metric in comparison.METRICS})
                        row["observed_spikes_per_unit"] = .4
                        row["unit_bin_mean_mse_corrected"] = -.2 if candidate else -.1
                        row["field_energy_score_hz"] = (1.5 if block else .5) if candidate else 1.
                        rows.append(row)
    table(root / "all_seeds.csv", rows)
    table(root / "training_costs.csv", [dict(seed=seed, parameter_count=2649,
          completed_epochs=17 if candidate else 20, best_epoch=12 if candidate else 17,
          hardware="{}", recorded_training_seconds=100.) for seed in (41, 42, 43)])
    write(root / "report.json", dict(
        status="COMPLETE_DESCRIPTIVE_PFC_POPULATION_FORECAST_ANALYSIS",
        training_seeds=[41, 42, 43], scientific_claim_qualified=False, formal_evaluation=False,
        preparation=str(prepared), preparation_sha256="", original_input_sha256="a" * 64,
        score_rows=1080, conditions=[f"condition{i}" for i in range(10)],
        validation_initial_noise_blocks=[0, 1, 2], validation_circuits=[74]))
    seal(root)


@pytest.fixture
def pair(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    make(reference, False)
    make(candidate, True)
    return reference, candidate, tmp_path / "comparison"


def change_rows(root, change):
    with (root / "all_seeds.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    change(rows)
    table(root / "all_seeds.csv", rows)
    seal(root)


def test_complete_pairs_and_signed_differences(pair):
    reference, candidate, output = pair
    result = comparison.compare(reference, candidate, output)
    assert result["paired_rows"] == 1080
    assert result["artifacts"]["by_source_condition_origin_horizon.csv"]["rows"] == 360
    assert result["artifacts"]["by_seed_horizon.csv"]["rows"] == 9
    assert result["rate_initialization_function_equal"] is False
    assert result["matched_forecast_random_streams"] is False
    assert result["control_geometry_causal_effect_identified"] is False
    with (output / "paired_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {float(r["delta_field_energy_score_hz"]) for r in rows} == {-.5, .5}
    assert all(float(r["delta_unit_bin_mean_mse_corrected"]) == pytest.approx(-.1) for r in rows)
    with (output / "by_horizon.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert all(float(r["delta_field_energy_score_hz_mean"]) == pytest.approx(1/6) for r in summary)
    assert all(int(r["paired_rows"]) == 360 for r in summary)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.__setitem__(-1, dict(rows[0])), "DUPLICATE_CONDITION"),
    (lambda rows: rows.pop(), "INCOMPLETE_CONDITION_GRID"),
    (lambda rows: rows[0].update(observed_spikes_per_unit=.5), "OBSERVATIONS_DIFFER"),
    (lambda rows: rows[0].update(field_energy_score_hz="nan"), "FINITE_REQUIRED"),
    (lambda rows: rows[0].update(origin_ms=101), "CLOCK_MISMATCH"),
])
def test_invalid_saved_rows_rejected(pair, change, match):
    reference, candidate, output = pair
    change_rows(candidate, change)
    with pytest.raises(ValueError, match=match):
        comparison.compare(reference, candidate, output)
    assert not output.exists()


def test_tampered_artifact_rejected(pair):
    reference, candidate, output = pair
    with (candidate / "all_seeds.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="BOUND_FILE_CHANGED"):
        comparison.compare(reference, candidate, output)


@pytest.mark.parametrize("kind,match", [
    ("inputs", "SOURCE_INPUTS_DIFFER"),
    ("helper", "SCORING_HELPER_DIFFERS"),
    ("protocol", "SCORING_PROTOCOL_DIFFERS"),
])
def test_changed_source_or_metric_contract_rejected(pair, kind, match):
    reference, candidate, output = pair
    prepared = candidate / "prepared"
    if kind == "inputs":
        path = prepared / "preparation.json"
        proof = json.loads(path.read_text())
        proof["input_files"][0]["sha256"] = "c" * 64
        write(path, proof)
    elif kind == "helper":
        (prepared / "Libs/population_dynamics/trajectory_scores.py").write_text("metric_definition = 2\n")
    else:
        path = prepared / "Libs/configs/neural_population_pfc_task_forecast.yaml"
        config = yaml.safe_load(path.read_text())
        config["joint_samples"] = 32
        path.write_text(yaml.safe_dump(config))
    seal(candidate)
    with pytest.raises(ValueError, match=match):
        comparison.compare(reference, candidate, output)
    assert not output.exists()


def test_existing_output_is_preserved(pair):
    reference, candidate, output = pair
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("authoritative result")
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        comparison.compare(reference, candidate, output)
    assert sentinel.read_text() == "authoritative result"


def shared_initializer_layout(root):
    source = root / "prepared/Libs/population_dynamics"
    old = (source / "pfc_population_training_data.py").read_text()
    (source / "source_population_training.py").write_text(old.replace("class Cohort:", "class SourcePopulationTrainingCohort:"))
    (source / "pfc_population_training_data.py").write_text(
        "from .source_population_training import SourcePopulationTrainingCohort\n"
        "class PfcPopulationTrainingCohort(SourcePopulationTrainingCohort):\n    pass\n")
    seal(root)


@pytest.mark.parametrize("migrate_reference", [False, True])
def test_initializer_reads_old_and_shared_delivered_layouts(pair, migrate_reference):
    reference, candidate, output = pair
    shared_initializer_layout(candidate)
    if migrate_reference:
        shared_initializer_layout(reference)
    result = comparison.compare(reference, candidate, output)
    assert result["rate_initialization_function_equal"] is False


def test_relocation_alone_preserves_initializer_equality(pair):
    reference, candidate, output = pair
    relative = "prepared/Libs/population_dynamics/pfc_population_training_data.py"
    (candidate / relative).write_text((reference / relative).read_text())
    shared_initializer_layout(candidate)
    result = comparison.compare(reference, candidate, output)
    assert result["rate_initialization_function_equal"] is True


def test_changed_shared_initializer_without_new_receipt_rejects(pair):
    reference, candidate, output = pair
    shared_initializer_layout(candidate)
    with (candidate / "prepared/Libs/population_dynamics/source_population_training.py").open("a") as stream:
        stream.write("# changed\n")
    with pytest.raises(ValueError, match="BOUND_FILE_CHANGED"):
        comparison.compare(reference, candidate, output)
    assert not output.exists()


def shared_evaluator_layout(root, layout):
    source = root / "prepared/Libs/population_dynamics"
    entry = source / "pfc_population_evaluation.py"
    shared = source / "source_population_evaluation.py"
    shared.write_text(entry.read_text())
    if layout == "import":
        entry.write_text("from .source_population_evaluation import "
                         + ", ".join(comparison.SCORING_FUNCTIONS) + "\n")
    else:
        entry.write_text("from . import source_population_evaluation as shared\n"
                         + "\n".join("def " + name + "():\n    return shared."
                                     + name + "()\n" for name in comparison.SCORING_FUNCTIONS))
    seal(root)
    return shared


@pytest.mark.parametrize("layout", ["import", "forward"])
@pytest.mark.parametrize("migrate_reference", [False, True])
def test_scoring_reads_inline_and_shared_numerical_implementations(pair, layout, migrate_reference):
    reference, candidate, output = pair
    shared = shared_evaluator_layout(candidate, layout)
    if migrate_reference:
        shared_evaluator_layout(reference, layout)
    result = comparison.compare(reference, candidate, output)
    assert result["scoring_functions_equal"] == list(comparison.SCORING_FUNCTIONS)
    assert result["consumed_inputs"]["candidate"][str(shared)] == hashlib.sha256(shared.read_bytes()).hexdigest()


@pytest.mark.parametrize("layout", ["import", "forward"])
def test_shared_evaluator_tampering_rejects_before_comparison_output(pair, layout):
    reference, candidate, output = pair
    shared = shared_evaluator_layout(candidate, layout)
    shared.write_text(shared.read_text() + "# Changed after the delivered binding.\n")
    with pytest.raises(ValueError, match="BOUND_FILE_CHANGED"):
        comparison.compare(reference, candidate, output)
    assert not output.exists()


@pytest.mark.parametrize("layout", ["import", "forward"])
@pytest.mark.parametrize("name", comparison.SCORING_FUNCTIONS)
def test_resealed_shared_numerical_change_rejects(pair, layout, name):
    reference, candidate, output = pair
    shared = shared_evaluator_layout(candidate, layout)
    original = "def " + name + "():\n    return 1\n"
    changed = "def " + name + "():\n    return 2\n"
    assert shared.read_text().count(original) == 1
    shared.write_text(shared.read_text().replace(original, changed))
    seal(candidate)
    with pytest.raises(ValueError, match="SCORING_FUNCTION_DIFFERS:" + name):
        comparison.compare(reference, candidate, output)
    assert not output.exists()

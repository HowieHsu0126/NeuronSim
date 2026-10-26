"""Synthetic contract tests only; these fixtures are not benchmark evidence."""
from __future__ import annotations

from copy import deepcopy

import pytest

from population_dynamics.appworld_benchmark_protocol import (
    aggregate_results,
    build_attempt_index,
    validate_split_manifest,
)


METHODS = {"neuronsim": {}, "official_react_code": {}}


def split_manifest():
    def split(prefix):
        # Deliberately interleaved: indexing must keep each scenario together.
        return {
            "task_ids": [f"{prefix}{scenario}_{variant}" for variant in (3, 1, 2) for scenario in ("a", "b")],
            "task_count": 6, "scenario_count": 2,
            # A fixture identity for source-file bytes, not a normalized-ID hash.
            "membership_sha256": "a" * 64,
        }
    return {
        "schema": "synthetic_appworld_split_fixture_v1", "appworld_version": "fixture-only",
        "data_root": "/synthetic/not-read", "splits": {"train": split("train"), "dev": split("dev")},
    }


def complete_records(manifest, splits=("train",)):
    return [{**attempt, "status": "completed", "native_tgc": True}
            for attempt in build_attempt_index(manifest, METHODS, splits)]


def test_complete_ordered_cohort_covers_every_task_method_seed():
    manifest = split_manifest()
    cohort = build_attempt_index(manifest, METHODS, ("train", "dev"))
    assert len(cohort) == 2 * 6 * 2 * 3
    assert [row["attempt_index"] for row in cohort] == list(range(len(cohort)))
    assert [(row["method"], row["seed"]) for row in cohort[:6]] == [
        (method, seed) for method in METHODS for seed in (41, 42, 43)
    ]
    assert [cohort[index]["task_id"] for index in range(0, 36, 6)] == [
        "traina_1", "traina_2", "traina_3", "trainb_1", "trainb_2", "trainb_3",
    ]
    assert manifest == split_manifest()


def test_native_tgc_sgc_and_arithmetic_seed_means_use_full_denominators():
    manifest = split_manifest()
    records = complete_records(manifest, ("train", "dev"))
    for row in records:
        if row["split"] == "train" and row["method"] == "neuronsim":
            if row["seed"] == 41 and row["task_id"] == "traina_2":
                row.update(status="agent_failure", native_tgc=False)
            if row["seed"] == 42:
                row["native_tgc"] = False
    report = aggregate_results(manifest, METHODS, ("train", "dev"), records)
    assert report["complete"] and report["expected_attempt_count"] == 72
    seed_rows = [row for row in report["by_seed"] if row["split"] == "train" and row["method"] == "neuronsim"]
    assert [row["tgc"] for row in seed_rows] == [5 / 6, 0, 1]
    assert [row["sgc"] for row in seed_rows] == [1 / 2, 0, 1]
    assert all(row["expected_task_count"] == row["scored_task_count"] == 6 for row in seed_rows)
    assert all(row["expected_scenario_count"] == row["scored_scenario_count"] == 2 for row in seed_rows)
    mean = report["seed_means"][0]
    assert mean["tgc_mean"] == pytest.approx((5 / 6 + 0 + 1) / 3)
    assert mean["sgc_mean"] == pytest.approx(0.5)
    assert report["pass_at_k"] == 1 and report["seeds"] == [41, 42, 43]
    assert len(report["agent_failures"]) == 1
    assert seed_rows[0]["status_counts"]["agent_failure"] == 1


def test_scored_native_success_survives_agent_failure():
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0]["status"] = "agent_failure"
    report = aggregate_results(manifest, METHODS, ("train",), records)
    assert report["complete"] and report["by_seed"][0]["tgc"] == 1
    assert report["agent_failures"][0]["native_tgc"] is True


@pytest.mark.parametrize("status", ["completed", "agent_failure", "infrastructure_failure", "evaluator_failure", "unknown"])
def test_unavailable_native_score_withholds_complete_metrics(status):
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0].update(status=status, native_tgc=None)
    report = aggregate_results(manifest, METHODS, ("train",), records)
    assert not report["complete"] and len(report["unscored_attempts"]) == 1
    affected = report["by_seed"][0]
    assert affected["expected_task_count"] == 6 and affected["scored_task_count"] == 5
    assert affected["scored_scenario_count"] == 1
    assert affected["tgc"] is None and affected["sgc"] is None
    assert report["seed_means"][0]["tgc_mean"] is None
    assert report["seed_means"][0]["sgc_mean"] is None
    assert report["seed_means"][1]["complete"]  # Other method is unaffected.


def test_missing_attempt_withholds_rates_and_retains_expected_denominator():
    manifest = split_manifest()
    records = complete_records(manifest)[1:]
    report = aggregate_results(manifest, METHODS, ("train",), records)
    assert not report["complete"]
    assert report["expected_attempt_count"] == 36 and report["observed_attempt_count"] == 35
    assert len(report["missing_attempts"]) == 1 and not report["unscored_attempts"]
    assert report["by_seed"][0]["missing_attempt_count"] == 1
    assert report["by_seed"][0]["expected_task_count"] == 6
    assert report["by_seed"][0]["tgc"] is None


def test_duplicate_attempt_rejected():
    manifest = split_manifest()
    records = complete_records(manifest)
    with pytest.raises(ValueError, match="duplicate terminal"):
        aggregate_results(manifest, METHODS, ("train",), [*records, records[0]])


@pytest.mark.parametrize("field,value", [("task_id", "unknown_1"), ("method", "unregistered"), ("split", "dev")])
def test_out_of_cohort_record_rejected(field, value):
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0][field] = value
    with pytest.raises(ValueError, match="out-of-cohort"):
        aggregate_results(manifest, METHODS, ("train",), records)


@pytest.mark.parametrize("seeds", [(41, 42), (41, 42, 44), (43, 42, 41), (41, 42, 43, 43), (41.0, 42, 43), None])
def test_seed_set_must_be_exact(seeds):
    with pytest.raises(ValueError, match="exactly"):
        build_attempt_index(split_manifest(), METHODS, ("train",), seeds)


@pytest.mark.parametrize("seed", [40, True, 41.0, "41"])
def test_record_seed_must_be_an_authorized_integer(seed):
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        aggregate_results(manifest, METHODS, ("train",), records)


@pytest.mark.parametrize("defect", ["duplicate", "missing_variant", "overlap", "task_count", "scenario_count", "digest", "tagged_line", "id_mapping"])
def test_invalid_manifest_membership_rejected(defect):
    manifest = deepcopy(split_manifest())
    train = manifest["splits"]["train"]
    if defect == "duplicate":
        train["task_ids"].append(train["task_ids"][0])
    elif defect == "missing_variant":
        train["task_ids"].pop()
        train["task_count"] -= 1
    elif defect == "overlap":
        manifest["splits"]["dev"] = deepcopy(train)
    elif defect in {"task_count", "scenario_count"}:
        train[defect] += 1
    elif defect == "digest":
        train["membership_sha256"] = "not-a-digest"
    elif defect == "tagged_line":
        train["task_ids"][0] += " : difficulty=1"
    elif defect == "id_mapping":
        train["task_ids"] = dict.fromkeys(train["task_ids"])
    with pytest.raises(ValueError):
        validate_split_manifest(manifest)


@pytest.mark.parametrize("native_tgc", [0, 1, "true", []])
def test_native_score_cannot_be_coerced(native_tgc):
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0]["native_tgc"] = native_tgc
    with pytest.raises(ValueError, match="bool or None"):
        aggregate_results(manifest, METHODS, ("train",), records)


def test_infrastructure_score_contradiction_is_rejected():
    manifest = split_manifest()
    records = complete_records(manifest)
    records[0]["status"] = "infrastructure_failure"
    with pytest.raises(ValueError, match="cannot carry"):
        aggregate_results(manifest, METHODS, ("train",), records)


def test_empty_results_do_not_create_a_zero_score():
    report = aggregate_results(split_manifest(), METHODS, ("train",), [])
    assert not report["complete"] and len(report["missing_attempts"]) == 36
    assert all(row["tgc"] is None and row["sgc"] is None for row in report["by_seed"])


def test_unknown_split_and_duplicate_methods_rejected():
    with pytest.raises(ValueError, match="unknown splits"):
        build_attempt_index(split_manifest(), METHODS, ("test_challenge",))
    with pytest.raises(ValueError, match="duplicates"):
        build_attempt_index(split_manifest(), ["neuronsim", "neuronsim"], ("train",))

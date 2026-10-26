"""Bounded AppWorld cohort indexing and complete, native-score aggregation.

This module consumes loader-resolved task IDs from an admitted split manifest.
It never parses native split-file lines or reads tasks, solutions or reports.
``membership_sha256`` identifies the exact native split-file bytes; the intake
and runner, not this module, verify those bytes and source admission.

The admitted original-benchmark contract has three variants, ``_1`` through
``_3``, per scenario. Validate that contract during native source admission.
Structural validation here does not admit a source or freeze a method.

A terminal record has ``split``, ``task_id``, ``method``, ``seed``, ``status`` and
``native_tgc``. Status is one of TERMINAL_STATUSES. ``native_tgc`` must be a bool
returned by the official evaluator, or None when that result is unavailable.
A scored agent failure can still be a task success: the native outcome wins.
Infrastructure, evaluator and unknown failures cannot carry a valid score.
Missing or unscored attempts withhold full-cohort metrics and seed means.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
from typing import Any


EXPERIMENT_SEEDS = (41, 42, 43)
TERMINAL_STATUSES = frozenset({
    "completed", "agent_failure", "infrastructure_failure", "evaluator_failure", "unknown",
})
_SCORABLE_STATUSES = frozenset({"completed", "agent_failure"})
_TASK_ID = re.compile(r"(?P<scenario>[^/\s]+)_(?P<variant>[123])")
_DIGEST = re.compile(r"[0-9a-fA-F]{64}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _names(values: Sequence[str] | Mapping[str, Any], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, Mapping)):
        raise ValueError(f"{label} must be an ordered sequence or mapping")
    names = tuple(_text(value, label) for value in values)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{label} must be nonempty and contain no duplicates")
    return names


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("seeds must be exactly [41, 42, 43] in that order")
    if any(type(seed) is not int for seed in seeds) or tuple(seeds) != EXPERIMENT_SEEDS:
        raise ValueError("seeds must be exactly [41, 42, 43] in that order")
    return EXPERIMENT_SEEDS


def _scenario(task_id: str) -> tuple[str, int]:
    match = _TASK_ID.fullmatch(_text(task_id, "task_id"))
    if match is None:
        raise ValueError(f"task_id must be a loader-resolved scenario_1/2/3 ID: {task_id!r}")
    return match.group("scenario"), int(match.group("variant"))


def validate_split_manifest(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Validate all split memberships and return IDs grouped by scenario.

    Keep scenarios in their first-seen native-loader order. Order their variants
    as 1, 2, 3. Require complete variants and disjoint tasks/scenarios across all
    manifest splits, including splits not selected for a particular run.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("split manifest must be a mapping")
    for field in ("schema", "appworld_version", "data_root"):
        _text(manifest.get(field), f"manifest.{field}")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("manifest.splits must be a nonempty mapping")
    memberships: dict[str, tuple[str, ...]] = {}
    seen_tasks: set[str] = set()
    seen_scenarios: set[str] = set()
    for split, entry in splits.items():
        _text(split, "split name")
        if not isinstance(entry, Mapping):
            raise ValueError(f"{split}: split entry must be a mapping")
        if not isinstance(entry.get("task_ids"), Sequence) or isinstance(entry["task_ids"], (str, bytes)):
            raise ValueError(f"{split}.task_ids must be a sequence of loader-resolved IDs")
        task_ids = _names(entry["task_ids"], f"{split}.task_ids")
        digest = entry.get("membership_sha256")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError(f"{split}: membership_sha256 must identify native split-file bytes")
        scenarios: dict[str, dict[int, str]] = {}
        for task_id in task_ids:
            if task_id in seen_tasks:
                raise ValueError(f"overlapping task ID across splits: {task_id}")
            scenario_id, variant = _scenario(task_id)
            scenarios.setdefault(scenario_id, {})[variant] = task_id
        for scenario_id, variants in scenarios.items():
            if set(variants) != {1, 2, 3}:
                raise ValueError(f"{split}: scenario {scenario_id} is missing required variants")
            if scenario_id in seen_scenarios:
                raise ValueError(f"overlapping scenario across splits: {scenario_id}")
        for field, expected in (("task_count", len(task_ids)), ("scenario_count", len(scenarios))):
            if type(entry.get(field)) is not int or entry[field] != expected:
                raise ValueError(f"{split}: {field} does not match admitted membership ({expected})")
        memberships[split] = tuple(
            variants[variant] for variants in scenarios.values() for variant in (1, 2, 3)
        )
        seen_tasks.update(task_ids)
        seen_scenarios.update(scenarios)
    return memberships


def build_attempt_index(
    manifest: Mapping[str, Any],
    methods: Sequence[str] | Mapping[str, Any],
    splits: Sequence[str],
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> list[dict[str, Any]]:
    """Return every task x registered method x seed, in deterministic order.

    A method mapping contributes every key in insertion order. Callers register
    controls in this same mapping/sequence. Method admission and freezing remain
    the runner's responsibility. Indices are local to this exact cohort.
    """
    memberships = validate_split_manifest(manifest)
    method_names = _names(methods, "methods")
    split_names = _names(splits, "splits")
    seed_values = _validate_seeds(seeds)
    unknown = set(split_names).difference(memberships)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    attempts: list[dict[str, Any]] = []
    for split in split_names:
        for task_id in memberships[split]:
            scenario_id, variant = _scenario(task_id)
            for method in method_names:
                for seed in seed_values:
                    attempts.append({
                        "attempt_index": len(attempts), "split": split, "task_id": task_id,
                        "scenario_id": scenario_id, "variant": variant, "method": method,
                        "seed": seed,
                    })
    return attempts


def _key(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    fields = tuple(_text(record.get(field), field) for field in ("split", "task_id", "method"))
    seed = record.get("seed")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("terminal record seed must be one of [41, 42, 43]")
    return (*fields, seed)


def aggregate_results(
    manifest: Mapping[str, Any],
    methods: Sequence[str] | Mapping[str, Any],
    splits: Sequence[str],
    outcomes: Sequence[Mapping[str, Any]],
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> dict[str, Any]:
    """Aggregate pass@1 TGC/SGC fractions in [0, 1] and arithmetic seed means.

    Reject duplicates and records outside the declared cohort. Preserve explicit
    agent failures, including failures after native success. Expected denominators
    always come from the manifest. A metric is None until its whole cohort has
    native scores. This function neither estimates significance nor pools seeds
    into independent scenarios; statistical procedures require a separate freeze.
    """
    attempts = build_attempt_index(manifest, methods, splits, seeds)
    expected = {_key(attempt): attempt for attempt in attempts}
    terminal: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    scored: dict[tuple[str, str, str, int], bool] = {}
    unscored: list[dict[str, Any]] = []
    agent_failures: list[dict[str, Any]] = []
    for record in outcomes:
        if not isinstance(record, Mapping):
            raise ValueError("each terminal record must be a mapping")
        key = _key(record)
        if key not in expected:
            raise ValueError(f"unknown or out-of-cohort task/method/split: {key}")
        if key in terminal:
            raise ValueError(f"duplicate terminal record: {key}")
        status = record.get("status")
        if not isinstance(status, str) or status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal status: {status!r}; use 'unknown' explicitly")
        if "native_tgc" not in record:
            raise ValueError(f"native_tgc is required; use None for an unavailable score: {key}")
        native_tgc = record["native_tgc"]
        if native_tgc is not None and type(native_tgc) is not bool:
            raise ValueError(f"native_tgc must be bool or None: {key}")
        if status not in _SCORABLE_STATUSES and native_tgc is not None:
            raise ValueError(f"{status} cannot carry a valid native_tgc: {key}")
        normalized = {**expected[key], "status": status, "native_tgc": native_tgc}
        terminal[key] = normalized
        if status == "agent_failure":
            agent_failures.append(normalized)
        if status in _SCORABLE_STATUSES and native_tgc is not None:
            scored[key] = native_tgc
        else:
            unscored.append(normalized)

    missing = [attempt for key, attempt in expected.items() if key not in terminal]
    by_seed: list[dict[str, Any]] = []
    seed_means: list[dict[str, Any]] = []
    for split in splits:
        for method in methods:
            method_rows: list[dict[str, Any]] = []
            for seed in seeds:
                cohort = [attempt for attempt in attempts if (
                    attempt["split"] == split and attempt["method"] == method and attempt["seed"] == seed
                )]
                keys = [_key(attempt) for attempt in cohort]
                status_counts = Counter(terminal[key]["status"] for key in keys if key in terminal)
                scenario_keys: dict[str, list[tuple[str, str, str, int]]] = {}
                for attempt, key in zip(cohort, keys):
                    scenario_keys.setdefault(attempt["scenario_id"], []).append(key)
                scenario_scores = [
                    all(scored[key] for key in group) for group in scenario_keys.values()
                    if all(key in scored for key in group)
                ]
                scored_count = sum(key in scored for key in keys)
                complete = scored_count == len(keys)
                task_successes = sum(scored[key] for key in keys if key in scored)
                row = {
                    "split": split, "method": method, "seed": seed, "complete": complete,
                    "expected_task_count": len(keys), "observed_attempt_count": sum(key in terminal for key in keys),
                    "scored_task_count": scored_count, "scored_task_success_count": task_successes,
                    "missing_attempt_count": sum(key not in terminal for key in keys),
                    "unscored_attempt_count": sum(key in terminal and key not in scored for key in keys),
                    "status_counts": {status: status_counts[status] for status in sorted(TERMINAL_STATUSES)},
                    "expected_scenario_count": len(scenario_keys),
                    "scored_scenario_count": len(scenario_scores),
                    "scored_scenario_success_count": sum(scenario_scores),
                    "tgc": task_successes / len(keys) if complete else None,
                    "sgc": sum(scenario_scores) / len(scenario_keys) if complete else None,
                }
                by_seed.append(row)
                method_rows.append(row)
            complete = all(row["complete"] for row in method_rows)
            seed_means.append({
                "split": split, "method": method, "complete": complete, "seed_count": len(seeds),
                "tgc_mean": sum(row["tgc"] for row in method_rows) / len(seeds) if complete else None,
                "sgc_mean": sum(row["sgc"] for row in method_rows) / len(seeds) if complete else None,
            })
    return {
        "schema": "appworld_benchmark_aggregation_v1", "appworld_version": manifest["appworld_version"],
        "source_manifest_schema": manifest["schema"], "seeds": list(seeds), "pass_at_k": 1,
        "metric_unit": "fraction",
        "methods": list(methods), "splits": list(splits), "complete": not missing and not unscored,
        "expected_attempt_count": len(attempts), "observed_attempt_count": len(terminal),
        "missing_attempts": missing, "unscored_attempts": unscored,
        "agent_failures": agent_failures, "by_seed": by_seed, "seed_means": seed_means,
    }

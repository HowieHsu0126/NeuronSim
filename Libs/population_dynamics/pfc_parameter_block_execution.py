"""Bind bounded PFC circuit/block source tasks to accepted input recipes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

CIRCUITS = (23, 63, 74, 86, 52)
DISTRACTORS = ("none", "aligned", "off_target", "antipodal", "late_off_target")
CONDITIONS = tuple(prefix + suffix for prefix in ("cue_", "omission_") for suffix in DISTRACTORS)
PREPARATION = "parameter_block_prepare_613107"
IDENTITY_INTS = ("circuit_index", "block_index", "condition_index", "connectivity_seed",
                 "initial_state_seed", "background_seed", "stimulus_seed")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(row: dict) -> dict:
    keys = ("scenario_id", "circuit_id", "circuit_index", "episode_id", "block_id",
            "block_index", "condition_index", "split", "connectivity_seed",
            "initial_state_seed", "background_seed", "stimulus_seed", "sample_unit")
    return {key: int(row[key]) if key in IDENTITY_INTS else row[key] for key in keys}


def recipe_path(row: dict, preparation: Path) -> Path:
    circuit, block, offset = (int(row[key]) for key in ("circuit_index", "block_index", "condition_index"))
    expected = f"recipes/circuit_{circuit:03d}/block_{block:03d}/{CONDITIONS[offset]}.yaml"
    if row["config_path"] != expected:
        raise ValueError("PFC_BLOCK_RECIPE_PATH_MISMATCH")
    path = preparation / expected
    if digest(path) != row["config_sha256"]:
        raise ValueError("PFC_BLOCK_RECIPE_HASH_MISMATCH")
    config = yaml.safe_load(path.read_text())
    if (config["benchmark_episode_candidate"]["identity"] != identity(row)
            or config["benchmark_episode_candidate"]["condition_id"] != row["condition_id"]
            or config["initial_rng_seed"] != int(row["initial_state_seed"])
            or config["external_noise_seed"] != int(row["background_seed"])
            or config["excitatory_neurons"] != 80000 or config["inhibitory_neurons"] != 20000
            or config["duration_ms"] != 5000.0 or config["dt_ms"] != 0.05
            or config["benchmark_episode_candidate"]["production_eligible"] is not False):
        raise ValueError("PFC_BLOCK_RECIPE_IDENTITY_OR_PROTOCOL_MISMATCH")
    return path


def load_plan(plan: Path, *, require_release: bool = False) -> list[dict]:
    delivery = json.loads((plan / "delivery.json").read_text())
    files = {row["path"]: row for row in delivery["files"]}
    if len(files) != len(delivery["files"]):
        raise ValueError("PFC_BLOCK_PLAN_DUPLICATE_FILE")
    binding = files["execution.csv"]
    if (delivery["status"] != "PREPARED_PFC_PARAMETER_BLOCK_EXECUTION"
            or digest(plan / "execution.csv") != binding["sha256"]
            or (plan / "execution.csv").stat().st_size != binding["bytes"]):
        raise ValueError("PFC_BLOCK_PLAN_BINDING_MISMATCH")
    with (plan / "execution.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    preparation = plan.parent / PREPARATION
    original_delivery = json.loads((preparation / "delivery.json").read_text())
    original_binding = next(row for row in original_delivery["files"] if row["path"] == "episodes.csv")
    if digest(preparation / "episodes.csv") != original_binding["sha256"]:
        raise ValueError("PFC_BLOCK_ORIGINAL_TABLE_BINDING_MISMATCH")
    with (preparation / "episodes.csv").open(newline="") as stream:
        original = list(csv.DictReader(stream))
    if len(rows) != len(original) or len(rows) != 100:
        raise ValueError("PFC_BLOCK_COMPLETE_ALLOCATION_REQUIRED")
    for index, (row, source) in enumerate(zip(rows, original)):
        if row != dict(source, task_index=str(index)):
            raise ValueError("PFC_BLOCK_PLAN_CHANGED_PREPARED_IDENTITY")
        circuit, block, condition = CIRCUITS[index // 20], (index % 20) // 10 + 1, index % 10
        if (int(row["circuit_index"]) != circuit or int(row["block_index"]) != block
                or int(row["condition_index"]) != condition or row["condition_id"] != CONDITIONS[condition]
                or int(row["initial_state_seed"]) != 210000000 + 1000 * circuit + block
                or int(row["background_seed"]) != 220000000 + 1000 * circuit + block):
            raise ValueError("PFC_BLOCK_CANONICAL_ALLOCATION_MISMATCH")
    if require_release:
        release = json.loads((plan / "source_release.json").read_text())
        if (release["status"] != "RELEASED_PFC_BOUNDED_PARAMETER_BLOCK_DEVELOPMENT"
                or release["source_execution_released"] is not True
                or release["production_eligible"] is not False
                or release["execution_sha256"] != binding["sha256"]
                or release["prerequisites"] != {
                    "four_existing_corner_delivery_chains_terminal_and_reconciled": True,
                    "interior_exact_tuple_native_input_and_stream_parity": True,
                    "declared_current_source_and_numerical_protocol": True}):
            raise ValueError("PFC_BLOCK_SOURCE_NOT_RELEASED")
        for receipt in release["receipts"]:
            path = Path(receipt["path"])
            if digest(path) != receipt["sha256"] or json.loads(path.read_text())["status"] != receipt["status"]:
                raise ValueError("PFC_BLOCK_RELEASE_RECEIPT_CHANGED")
    return rows


def group_rows(rows: list[dict], group: int) -> list[dict]:
    if not 0 <= group < 10:
        raise ValueError("PFC_BLOCK_GROUP_OUT_OF_RANGE")
    selected = rows[group * 10:(group + 1) * 10]
    if (len(selected) != 10 or [row["condition_id"] for row in selected] != list(CONDITIONS)
            or len({(row["circuit_index"], row["block_index"]) for row in selected}) != 1):
        raise ValueError("PFC_BLOCK_COMPLETE_CONDITION_GROUP_REQUIRED")
    return selected


def source_path(base: Path, row: dict) -> Path:
    return (base / "parameter_block_source" / f"circuit_{int(row['circuit_index']):03d}"
            / f"block_{int(row['block_index']):03d}" / row["condition_id"]
            / f"index_{int(row['task_index']):03d}")


def complete_source_batch(base: Path, task: dict) -> tuple[Path, list[dict]]:
    """Reuse one complete execution while retaining failed resource attempts.

    A second successful execution is ambiguous and must not select evidence.
    Pending, foreign and partial-success attempts also prevent admission.
    """
    source = source_path(base, task)
    batches = source / "batches"
    attempts, successful = [], []
    for path in sorted(batches.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("PFC_BLOCK_INVALID_SOURCE_ATTEMPT_PATH")
        receipt = path / "delivery.json"
        plan = json.loads(receipt.read_text())
        job = plan.get("job_id")
        cases = plan.get("cases")
        if (not isinstance(job, str) or not job.isdecimal() or path.name != "run_" + job
                or not isinstance(cases, list) or len(cases) != 1
                or plan.get("case_ids") != [task["episode_id"]]
                or plan.get("input_bindings", {}).get("actual", {}).get("config_sha256")
                != task["config_sha256"]):
            raise ValueError("PFC_BLOCK_SOURCE_ATTEMPT_IDENTITY_MISMATCH")
        case = cases[0]
        if (case.get("job_id") != job or case.get("case_id") != task["episode_id"]
                or case.get("node") != "cpu03" or case.get("archive_status") != "COMPLETE"
                or Path(case.get("archive", "")) != source / ("run_" + job)
                or case.get("config_sha256") != task["config_sha256"]):
            raise ValueError("PFC_BLOCK_SOURCE_ATTEMPT_ARCHIVE_MISMATCH")
        complete = (plan.get("status") == "COMPLETE"
                    and plan.get("exit_code") == plan.get("srun_exit_code") == 0
                    and case.get("exit_code") == 0
                    and case.get("source_execution_complete") is True)
        failed = (plan.get("status") == "FAILED"
                  and type(plan.get("exit_code")) is int and plan["exit_code"] != 0
                  and type(plan.get("srun_exit_code")) is int and plan["srun_exit_code"] != 0
                  and type(case.get("exit_code")) is int and case["exit_code"] != 0
                  and case.get("source_execution_complete") is False)
        if not complete and not failed:
            raise ValueError("PFC_BLOCK_SOURCE_ATTEMPT_NOT_TERMINAL")
        attempts.append(dict(job_id=job, status=plan["status"], exit_code=plan["exit_code"],
                             source_execution_complete=case["source_execution_complete"],
                             receipt=str(receipt), receipt_sha256=digest(receipt)))
        if complete:
            successful.append(path)
    if len(successful) != 1:
        raise ValueError("PFC_BLOCK_EXACTLY_ONE_COMPLETE_SOURCE_ATTEMPT_REQUIRED")
    return successful[0], attempts


def qualification_path(base: Path, circuit: int) -> Path:
    if circuit == 52:
        return base / "interior_native_parity/stream_numpy_613135"
    points = {23: "sigma30_jpos2p3", 63: "sigma20_jpos2p3", 86: "sigma20_jpos1p6_default"}
    if circuit == 74:
        return base / "missing_boundary_native/stream_numpy_611196"
    return base / "boundary_parity_610557" / points[circuit]

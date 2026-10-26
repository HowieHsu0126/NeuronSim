"""Source attempt admission after interrupted PFC development execution."""
import json
from pathlib import Path

import pytest

from population_dynamics.pfc_parameter_block_execution import complete_source_batch, source_path


def task():
    return dict(circuit_index="23", block_index="1", condition_id="omission_off_target",
                task_index="7", episode_id="pfc_working_memory_circuit_023_block_001_condition_07",
                config_sha256="a" * 64)


def attempt(base: Path, row: dict, job: str, *, success=True):
    source = source_path(base, row)
    path = source / "batches" / ("run_" + job)
    path.mkdir(parents=True)
    code = 0 if success else 1
    value = dict(job_id=job, status="COMPLETE" if success else "FAILED", exit_code=code,
                 srun_exit_code=code, case_ids=[row["episode_id"]],
                 input_bindings=dict(actual=dict(config_sha256=row["config_sha256"])),
                 cases=[dict(job_id=job, case_id=row["episode_id"], node="cpu03",
                             archive_status="COMPLETE", archive=str(source / ("run_" + job)),
                             config_sha256=row["config_sha256"], exit_code=code,
                             source_execution_complete=success)])
    (path / "delivery.json").write_text(json.dumps(value))
    return path, value


def test_original_single_complete_execution(tmp_path):
    row = task()
    path, _ = attempt(tmp_path, row, "100")
    selected, observations = complete_source_batch(tmp_path, row)
    assert selected == path and len(observations) == 1
    assert observations[0]["source_execution_complete"] is True


def test_failed_resource_attempt_preserved_beside_complete_retry(tmp_path):
    row = task()
    failed, _ = attempt(tmp_path, row, "100", success=False)
    original = (failed / "delivery.json").read_bytes()
    recovered, _ = attempt(tmp_path, row, "200")
    selected, observations = complete_source_batch(tmp_path, row)
    assert selected == recovered
    assert [value["status"] for value in observations] == ["FAILED", "COMPLETE"]
    assert (failed / "delivery.json").read_bytes() == original


def test_two_successful_attempts_reject(tmp_path):
    row = task()
    attempt(tmp_path, row, "100")
    attempt(tmp_path, row, "200")
    with pytest.raises(ValueError, match="EXACTLY_ONE_COMPLETE"):
        complete_source_batch(tmp_path, row)


def test_failed_only_rejects(tmp_path):
    row = task()
    attempt(tmp_path, row, "100", success=False)
    with pytest.raises(ValueError, match="EXACTLY_ONE_COMPLETE"):
        complete_source_batch(tmp_path, row)


@pytest.mark.parametrize("key,value", [("case_ids", ["foreign"]),
    ("input_bindings", {"actual": {"config_sha256": "b" * 64}}), ("status", "RUNNING")])
def test_foreign_or_nonterminal_attempt_rejects(tmp_path, key, value):
    row = task()
    path, record = attempt(tmp_path, row, "100")
    record[key] = value
    (path / "delivery.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH|NOT_TERMINAL"):
        complete_source_batch(tmp_path, row)

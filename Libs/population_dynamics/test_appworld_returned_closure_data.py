"""Source-outcome admission for complete manifest-bound CPU closure returns."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_appworld_returned_closure_data.py"))["verify_return_source"]


def returned_case(failed):
    code, status = (1, "FAILED") if failed else (0, "PASS")
    outcome = "failed_diagnostics" if failed else "completed"
    identity = {"bytes": 123, "sha256": "synthetic-fixture-sha256"}
    source = {"run_id": "synthetic-source", "trial_id": "synthetic-trial", "exit_code": code, "status": status, "persisted": True}
    receipt = {
        "status": "PASS_FAILED_SOURCE_DIAGNOSTICS" if failed else "PASS",
        "exit_code": 0, "persisted": True,
        "source_run_id": source["run_id"], "source_trial_id": source["trial_id"],
        "source_exit_code": code, "source_status": status, "source_outcome": outcome,
        "returned_files": identity,
    }
    returned = {"source_run_id": source["run_id"], "source_trial_id": source["trial_id"]}
    return {"source_receipt": source, "source_outcome": outcome}, returned, receipt, deepcopy(identity)


@pytest.mark.parametrize("failed", [False, True])
def test_complete_return_retains_source_exit_and_status(failed):
    args = returned_case(failed)
    original = deepcopy(args)
    admitted = VERIFY(*args)
    assert admitted == args[0]["source_receipt"]
    assert admitted["exit_code"] == (1 if failed else 0)
    assert args == original


def test_existing_completed_receipt_without_diagnostic_fields_remains_supported():
    intake, returned, receipt, identity = returned_case(False)
    intake.pop("source_outcome")
    for key in ("source_trial_id", "source_status", "source_exit_code", "source_outcome"):
        receipt.pop(key)
    assert VERIFY(intake, returned, receipt, identity)["exit_code"] == 0


@pytest.mark.parametrize("location,key,value", [
    ("intake", "source_outcome", "completed"),
    ("intake", "source_outcome", "unknown"),
    ("source", "persisted", False),
    ("source", "exit_code", True),
    ("receipt", "status", "PASS"),
    ("receipt", "exit_code", 1),
    ("receipt", "persisted", False),
    ("receipt", "source_exit_code", 0),
    ("receipt", "source_status", "PASS"),
    ("receipt", "source_trial_id", "another-trial"),
    ("receipt", "source_run_id", "another-run"),
    ("receipt", "source_outcome", "completed"),
    ("returned", "source_run_id", "another-run"),
    ("returned", "source_trial_id", "another-trial"),
    ("identity", "bytes", 124),
    ("identity", "sha256", "different"),
])
def test_failed_source_cannot_be_promoted_or_mismatched(location, key, value):
    intake, returned, receipt, identity = returned_case(True)
    targets = {"intake": intake, "source": intake["source_receipt"], "returned": returned, "receipt": receipt, "identity": identity}
    targets[location][key] = value
    with pytest.raises(ValueError):
        VERIFY(intake, returned, receipt, identity)

"""Mock-native scoring tests only; fixtures are not benchmark evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from population_dynamics import appworld_benchmark_scoring as scoring


@pytest.fixture
def scoring_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    data = source_root / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "tasks").mkdir()
    manifest = {"schema": "appworld_split_membership_v1", "appworld_version": "fixture-version", "data_root": str(source_root), "splits": {}}
    for split, prefix in (("train", "traincase"), ("dev", "devcase"), ("test_challenge", "challengecase")):
        ids = [f"{prefix}_{variant}" for variant in (1, 2, 3)]
        payload = (ids[0] + ":tag\n" + "\n".join(ids[1:]) + "\n").encode()
        membership_file = data / "datasets" / f"{split}.txt"
        membership_file.write_bytes(payload)
        manifest["splits"][split] = {
            "task_ids": ids, "task_count": 3, "scenario_count": 1,
            "membership_file": str(membership_file), "membership_sha256": hashlib.sha256(payload).hexdigest(),
        }
    root = tmp_path / "execution"
    root.mkdir()
    (root / "data").symlink_to(data, target_is_directory=True)
    receipt = {
        "schema": "appworld_development_execution_v1", "scope": "development_only", "formal_eligible": False,
        "split": "dev", "task_id": "devcase_1", "method": "official_react_code", "seed": 41,
        "terminal": True, "status": "completed", "native_tgc": None, "scoring_status": "not_run",
        "usage_complete": False, "appworld_version": "fixture-version", "data_root": str(source_root),
        "run_root": str(root), "experiment_name": "development", "official_solve_returned": True,
    }
    receipt_path = root / "execution_receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt_path.write_text(json.dumps(receipt))
    manifest_path.write_text(json.dumps(manifest))
    evaluation_dir = root / "experiments" / "outputs" / "development" / "tasks" / "devcase_1" / "evaluation"
    state = SimpleNamespace(
        manifest=manifest, receipt=receipt, root=root, output=tmp_path / "score.json",
        receipt_path=receipt_path, manifest_path=manifest_path, evaluation_dir=evaluation_dir,
        report=evaluation_dir / "report.md", version_file=evaluation_dir / "version.txt",
        original_receipt=receipt_path.read_bytes(), original_manifest=manifest_path.read_bytes(),
        load_calls=[], evaluate_calls=[], evaluator_error=None, retain_reports=True,
        result={
            "aggregate": {"task_goal_completion": 0, "scenario_goal_completion": 0},
            "individual": {"devcase_1": {"success": True, "difficulty": 1, "num_tests": 2, "passes": [{"requirement": "a"}, {"requirement": "b"}], "failures": []}},
        },
    )

    class FakePathStore:
        root = str(tmp_path / "prior-root")

        def update_root(self, root):
            self.root = root
            os.environ["APPWORLD_ROOT"] = root

        def reload(self):
            self.root = os.environ.get("APPWORLD_ROOT", str(tmp_path / "prior-root"))

    def load_task_ids(**kwargs):
        assert set(kwargs) == {"dataset_name"}
        assert Path(os.environ["APPWORLD_ROOT"], "data").resolve() == data
        state.load_calls.append(kwargs)
        return list(manifest["splits"][kwargs["dataset_name"]]["task_ids"])

    def evaluate_tasks(**kwargs):
        state.evaluate_calls.append(kwargs)
        assert os.environ["APPWORLD_ROOT"] == str(root)
        if state.evaluator_error is not None:
            raise state.evaluator_error
        if state.retain_reports:
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            state.report.write_text("Synthetic native report fixture")
            state.version_file.write_text("fixture-version")
        return deepcopy(state.result)

    state.native = SimpleNamespace(version="fixture-version", path_store=FakePathStore(), load_task_ids=load_task_ids)
    monkeypatch.setattr(scoring, "_load_native", lambda: state.native)
    monkeypatch.setattr(scoring, "_load_evaluator", lambda: evaluate_tasks)
    monkeypatch.setattr(scoring, "_check_execution_location", lambda root: None)
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path / "prior-root"))
    monkeypatch.setenv("APPWORLD_CACHE", str(tmp_path / "prior-cache"))
    return state


def score(fixture):
    return scoring.score_development_attempt(
        fixture.receipt, fixture.manifest, fixture.output,
        execution_receipt_path=fixture.receipt_path, split_manifest_path=fixture.manifest_path,
    )


def test_positive_uses_exact_native_task_row_and_retains_provenance(scoring_fixture):
    fixture = scoring_fixture
    saved_environment = {key: os.environ.get(key) for key in ("APPWORLD_ROOT", "APPWORLD_CACHE")}
    outcome = score(fixture)
    assert outcome["native_tgc"] is True and outcome["status"] == "completed"
    assert outcome["scoring_status"] == "completed" and not outcome["usage_complete"]
    assert outcome["membership_verified"] and outcome["evaluator_called"]
    assert fixture.evaluate_calls == [{
        "task_ids": ["devcase_1"], "experiment_name": "development",
        "suppress_errors": True, "include_details": True, "save_reports": True,
    }]
    assert outcome["native_task_result"] == fixture.result["individual"]["devcase_1"]
    assert "aggregate" not in outcome and "sgc" not in outcome
    assert outcome["native_evaluation_report"] == str(fixture.report)
    assert outcome["provenance"]["execution_receipt_path"] == str(fixture.receipt_path)
    assert fixture.receipt_path.read_bytes() == fixture.original_receipt
    assert fixture.manifest_path.read_bytes() == fixture.original_manifest
    assert json.loads(fixture.output.read_text()) == outcome
    assert {key: os.environ.get(key) for key in saved_environment} == saved_environment


def test_incomplete_native_negative_remains_false_without_invented_checks(scoring_fixture):
    fixture = scoring_fixture
    row = fixture.result["individual"]["devcase_1"]
    row.update(success=False, num_tests=3, passes=[{"requirement": "a"}], failures=[])
    outcome = score(fixture)
    assert outcome["native_tgc"] is False and outcome["scoring_status"] == "completed"
    assert outcome["native_task_result"]["num_tests"] == 3
    assert outcome["native_task_result"]["passes"] == [{"requirement": "a"}]
    assert outcome["native_task_result"]["failures"] == []


def test_native_success_preserves_agent_failure_execution_status(scoring_fixture):
    scoring_fixture.receipt["status"] = "agent_failure"
    outcome = score(scoring_fixture)
    assert outcome["status"] == "agent_failure" and outcome["execution_status"] == "agent_failure"
    assert outcome["native_tgc"] is True and outcome["scoring_status"] == "completed"


@pytest.mark.parametrize("status", ["infrastructure_failure", "evaluator_failure", "unknown"])
def test_other_terminal_statuses_remain_unscored_without_native_access(scoring_fixture, monkeypatch, status):
    fixture = scoring_fixture
    fixture.receipt["status"] = status
    monkeypatch.setattr(scoring, "_load_native", lambda: pytest.fail("unscored outcomes must not import native code"))
    monkeypatch.setattr(scoring, "_load_evaluator", lambda: pytest.fail("unscored outcomes must not access the evaluator"))
    outcome = score(fixture)
    assert outcome["status"] == status and outcome["native_tgc"] is None
    assert outcome["scoring_status"] == "not_run" and not outcome["evaluator_called"]
    assert outcome["usage_complete"] is False and not fixture.load_calls


def test_evaluator_exception_is_explicit_unscored_failure(scoring_fixture):
    fixture = scoring_fixture
    fixture.evaluator_error = RuntimeError("synthetic native evaluator failure")
    outcome = score(fixture)
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None
    assert outcome["scoring_status"] == "evaluator_failure" and outcome["evaluator_called"]
    assert outcome["error_type"] == "RuntimeError"
    assert fixture.receipt_path.read_bytes() == fixture.original_receipt


def test_native_task_result_mismatch_is_rejected(scoring_fixture):
    fixture = scoring_fixture
    fixture.result["individual"] = {"devcase_2": fixture.result["individual"]["devcase_1"]}
    outcome = score(fixture)
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None
    assert "exactly the requested task" in outcome["error"]
    assert fixture.report.is_file()  # Preserve even an unusable evaluator result.


def test_receipt_task_mismatch_is_rejected_before_evaluation(scoring_fixture):
    fixture = scoring_fixture
    fixture.receipt["task_id"] = "absent_1"
    with pytest.raises(ValueError, match="split membership"):
        score(fixture)
    assert not fixture.evaluate_calls and not fixture.output.exists()


def test_formal_split_is_rejected_without_native_access(scoring_fixture, monkeypatch):
    fixture = scoring_fixture
    fixture.receipt.update(split="test_challenge", task_id="challengecase_1")
    monkeypatch.setattr(scoring, "_load_native", lambda: pytest.fail("test split must remain closed"))
    with pytest.raises(ValueError, match="formal/test"):
        score(fixture)
    assert not fixture.output.exists()


@pytest.mark.parametrize("defect", ["zero_tests", "boolean_tests", "non_boolean_success", "missing_details", "incomplete_positive", "too_many_checks"])
def test_malformed_native_checks_are_not_coerced(scoring_fixture, defect):
    fixture = scoring_fixture
    row = fixture.result["individual"]["devcase_1"]
    if defect == "zero_tests":
        row["num_tests"] = 0
    elif defect == "boolean_tests":
        row["num_tests"] = True
    elif defect == "non_boolean_success":
        row["success"] = 1
    elif defect == "missing_details":
        row.pop("passes")
    elif defect == "incomplete_positive":
        row["passes"].pop()
    elif defect == "too_many_checks":
        row["failures"] = [{"requirement": "extra"}]
    outcome = score(fixture)
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None


@pytest.mark.parametrize("field", ["report", "version_file"])
def test_existing_native_reports_are_never_overwritten(scoring_fixture, field):
    fixture = scoring_fixture
    path = getattr(fixture, field)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserved prior evidence")
    outcome = score(fixture)
    assert outcome["status"] == "evaluator_failure" and not outcome["evaluator_called"]
    assert path.read_bytes() == b"preserved prior evidence" and not fixture.evaluate_calls


def test_missing_native_reports_withhold_score(scoring_fixture):
    scoring_fixture.retain_reports = False
    outcome = score(scoring_fixture)
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None


def test_existing_score_output_is_preserved_before_native_access(scoring_fixture):
    fixture = scoring_fixture
    fixture.output.write_bytes(b"existing or unresolved score output")
    with pytest.raises(FileExistsError):
        score(fixture)
    assert fixture.output.read_bytes() == b"existing or unresolved score output"
    assert not fixture.load_calls and not fixture.evaluate_calls


def test_execution_receipt_cannot_be_used_as_score_output(scoring_fixture):
    fixture = scoring_fixture
    with pytest.raises(FileExistsError):
        scoring.score_development_attempt(fixture.receipt, fixture.manifest, fixture.receipt_path)
    assert fixture.receipt_path.read_bytes() == fixture.original_receipt
    assert not fixture.evaluate_calls


@pytest.mark.parametrize("defect", ["version", "membership", "data_link"])
def test_native_admission_identity_mismatch_withholds_scoring(scoring_fixture, defect):
    fixture = scoring_fixture
    if defect == "version":
        fixture.native.version = "unadmitted-version"
    elif defect == "membership":
        Path(fixture.manifest["splits"]["dev"]["membership_file"]).write_text("changed native membership")
    elif defect == "data_link":
        (fixture.root / "data").unlink()
        (fixture.root / "data").symlink_to(fixture.root.parent)
    outcome = score(fixture)
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None
    assert not fixture.evaluate_calls


@pytest.mark.parametrize("success", [True, False])
def test_cli_returns_zero_for_valid_positive_and_negative(scoring_fixture, success):
    fixture = scoring_fixture
    row = fixture.result["individual"]["devcase_1"]
    if not success:
        row.update(success=False, passes=[{"requirement": "a"}], failures=[{"requirement": "b"}])
    exit_code = scoring.main([
        "--execution-receipt", str(fixture.receipt_path), "--split-manifest", str(fixture.manifest_path),
        "--output", str(fixture.output),
    ])
    assert exit_code == 0
    assert json.loads(fixture.output.read_text())["native_tgc"] is success


def test_cli_returns_nonzero_for_evaluator_failure(scoring_fixture):
    fixture = scoring_fixture
    fixture.evaluator_error = RuntimeError("synthetic evaluator error")
    exit_code = scoring.main([
        "--execution-receipt", str(fixture.receipt_path), "--split-manifest", str(fixture.manifest_path),
        "--output", str(fixture.output),
    ])
    assert exit_code != 0
    outcome = json.loads(fixture.output.read_text())
    assert outcome["status"] == "evaluator_failure" and outcome["native_tgc"] is None

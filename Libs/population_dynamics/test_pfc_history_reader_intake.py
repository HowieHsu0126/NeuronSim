"""Historical reader intake preserves logical group and actual execution identity."""
import ast
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def validate():
    script = Path(__file__).resolve().parents[1] / "scripts/intake_pfc_completed_history_pair.sbatch"
    code = script.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    node = next(node for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef) and node.name == "validate_reader_execution")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["validate_reader_execution"]


@pytest.mark.parametrize("job,index,step", [
    ("610296_43", 43, None),
    ("615390_45", 45, None),
    ("615390_45", 45, "batch"),
    ("612224", 42, "253"),
])
def test_actual_execution_is_accepted(validate, job, index, step):
    report = dict(job_id=job, slurm_step_id=step)
    before = dict(report)
    validate(report, job, index, step)
    assert report == before


@pytest.mark.parametrize("job,index,report_job,report_step,expected_step", [
    ("615390_45", 44, "615390_45", None, None),
    ("615390_45", 45, "610296_45", None, None),
    ("615390_45", 45, "615390_45", "253", None),
    ("0_45", 45, "0_45", None, None),
    ("615390_45_1", 45, "615390_45_1", None, None),
    ("615390_x", 45, "615390_x", None, None),
    ("612224", 42, "612224", "253", "254"),
    ("612224", 42, "612224", None, None),
    ("invalid", 42, "invalid", "253", "253"),
])
def test_execution_mismatch_rejects(validate, job, index, report_job, report_step, expected_step):
    report = dict(job_id=report_job, slurm_step_id=report_step)
    with pytest.raises(AssertionError):
        validate(report, job, index, expected_step)

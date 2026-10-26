"""Source-history admission checks the distinct native and refined-clock contracts."""
import ast
import copy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def validator():
    script = Path(__file__).resolve().parents[1] / "scripts/intake_pfc_completed_fine_history.sbatch"
    text = script.read_text().split("<<'PYCODE'\n", 1)[1].rsplit("\nPYCODE", 1)[0]
    tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_history_source")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["validate_history_source"]


@pytest.fixture
def native():
    totals = {"E": 560008601, "I": 139986425}
    original = dict(status="PFC_SOURCE_EXECUTION_COMPLETE", source_execution_complete=True,
        seeds={"source_rng_seed": 95013}, source_entry_sha256="source_hash", random_stream_policy="source_global",
        dimensions=dict(coupled_core_units=100000, n_e=80000, n_i=20000, dt_seconds=.00005,
                        duration_seconds=5, duration_ticks=100000), completed_duration_ticks=100000,
        inputs_used=dict(input_mode="native_numpy_stream", ticks_consumed=100000,
                         E=dict(events=totals["E"], values=8000000000),
                         I=dict(events=totals["I"], values=2000000000)),
        source_external_event_totals=dict(totals), validations=dict(
            canonical_event_schema=dict(passed=True, event_count=4936749),
            complete_source_input_consumption=dict(passed=True, event_totals=dict(totals))))
    analysis = dict(reconstruction=dict(ticks_reconstructed=100000, source_events_consumed=4936749),
        reconstruction_drive=dict(seeds={"source_rng_seed": 95013}, random_stream_policy="source_global",
                                  source_entry_sha256="source_hash", ticks_generated=100000,
                                  dt_seconds=.00005, external_event_totals=dict(totals)))
    return original, dict(refinement_seeds=None), analysis


def test_valid_native_preserves_records(validator, native):
    before = copy.deepcopy(native)
    validator(*native, "native", 95013)
    assert native == before


@pytest.mark.parametrize("path,value", [
    ((0,"seeds","source_rng_seed"), 95014),
    ((0,"status"), "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"),
    ((0,"dimensions","dt_seconds"), .000025),
    ((0,"dimensions","n_e"), 79999),
    ((0,"inputs_used","E","values"), 7999999999),
    ((0,"inputs_used","input_mode"), "coupled_binomial_clock_refinement_stream"),
    ((0,"source_external_event_totals","E"), 1),
    ((1,"refinement_seeds"), [1,2]),
    ((2,"reconstruction_drive","source_entry_sha256"), "wrong"),
    ((2,"reconstruction_drive","seeds","source_rng_seed"), 95014),
    ((2,"reconstruction_drive","external_event_totals","I"), 1),
    ((2,"reconstruction","ticks_reconstructed"), 99999),
    ((2,"reconstruction","source_events_consumed"), 1),
    ((0,"validations","canonical_event_schema","passed"), False),
])
def test_native_rejects_changed_source_contract(validator, native, path, value):
    target = native[path[0]]
    for key in path[1:-1]:
        target = target[key]
    target[path[-1]] = value
    before = copy.deepcopy(native)
    with pytest.raises(AssertionError):
        validator(*native, "native", 95013)
    assert native == before


@pytest.mark.parametrize("label,dt", [("half_A",.025),("half_B",.025),
                                      ("quarter_A",.0125),("quarter_B",.0125)])
def test_refined_clock_contract_still_accepts(validator, native, label, dt):
    original, binding, analysis = native
    ticks = round(5000/dt)
    original.update(status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE", coupled_core_units=100000,
                    source_rng_seed=95013, dt_ms=dt, duration_ms=5000,
                    completed_duration_ticks=ticks, refinement_seeds=[1,2],
                    drive=dict(dt_seconds=dt/1000, external_event_totals=original["source_external_event_totals"]))
    binding["refinement_seeds"] = [1,2]
    analysis["reconstruction"]["ticks_reconstructed"] = ticks
    analysis["reconstruction_drive"].update(ticks_generated=ticks, dt_seconds=dt/1000,
        coupling=dict(coarse_source=dict(seeds=dict(source_rng_seed=95013)), refinement_seeds=[1,2]))
    before = copy.deepcopy(native)
    validator(*native, label, 95013)
    assert native == before

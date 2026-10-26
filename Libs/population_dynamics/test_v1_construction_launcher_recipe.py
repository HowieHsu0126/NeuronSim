"""Test realization changes and recipe rejection at the actual launcher boundary."""
import ast
import copy
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ("run_v1_recurrent_construction.sbatch", "run_v1_input_construction.sbatch"))
def test_recipe_binding_admits_independent_seeds_and_rejects_recipe_changes(name):
    launcher = Path(__file__).parents[1] / "scripts" / name
    source = launcher.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "realization_keys"
                           for target in node.targets)]
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(isinstance(value, ast.Constant) and isinstance(value.value, str)
                      and "RECIPE_CHANGED_SINCE_CANARY" in value.value for value in ast.walk(node))]
    assert len(assignments) == len(checks) == 1
    code = compile(ast.Module(body=[assignments[0], checks[0]], type_ignores=[]), str(launcher), "exec")
    original = dict(schema_version=1, source_root="/canary/sources",
                    bmtk_wheel=dict(path="/canary/bmtk.whl", sha256="pinned-wheel"),
                    builder_seed=100000000, glif_assignment_seed=101000000,
                    expected_full_nodes=230924, canary_fraction=0.01, full_fraction=1.0,
                    edge_chunk_size=1000000, benchmark_circuit_candidate=dict(circuit_index=0))
    config = copy.deepcopy(original)
    config.update(builder_seed=100000001, glif_assignment_seed=101000001,
                  benchmark_circuit_candidate=dict(circuit_index=1), source_root="/current/sources")
    config["bmtk_wheel"]["path"] = "/current/bmtk.whl"
    exec(code, dict(config=config, canary={"original_config": original}))
    for key, value in (("expected_full_nodes", 1000), ("full_fraction", 0.1),
                       ("edge_chunk_size", 123), ("undeclared_recipe_option", True)):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="RECIPE_CHANGED_SINCE_CANARY"):
            exec(code, dict(config=changed, canary={"original_config": original}))
    changed = copy.deepcopy(config)
    changed["bmtk_wheel"]["sha256"] = "different-wheel"
    with pytest.raises(ValueError, match="RECIPE_CHANGED_SINCE_CANARY"):
        exec(code, dict(config=changed, canary={"original_config": original}))

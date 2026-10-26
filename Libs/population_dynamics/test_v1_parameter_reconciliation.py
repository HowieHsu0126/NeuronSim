"""Source joins, precision boundaries and receipt-bound parameter comparison."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest

from . import v1_parameter_reconciliation as reconciliation
from .provenance import sha256_file
from .test_v1_graph_identity import archive_fixture, node_group
from .v1_cell_parameters import BMTK_CONVERTER_SHA256, GLIF3_TEMPLATE
from .v1_graph_identity import inspect_archive


def reference(identity, params=None):
    original = node_group([0])["params"] if params is None else params
    converted = {reconciliation.REFERENCE_NAMES[key]: deepcopy(value) for key, value in original.items()}
    converted.update(after_spike_currents=True, spike_dependent_threshold=False,
                     adapting_threshold=False, V_m=-65.0, asc_init=[0.0, 0.0])
    return dict(specimen_id=identity, neuronal_model_id=identity + 100,
                neuronal_model_template_id=GLIF3_TEMPLATE,
                converted_template="nest:glif_psc", converted_parameters=converted)


def test_join_uses_source_node_ids_with_reordered_groups_and_reference_rows():
    nodes = [node_group([2]), node_group([1, 0])]
    nodes[0]["params"]["k"] = [0.3, 0.5]
    report = reconciliation.compare_parameters(nodes, np.array([10, 10, 20]),
        [reference(10), reference(20, nodes[0]["params"])])
    assert [row["specimen_id"] for row in report["cells"]] == [20, 10]
    assert report["loader_parameters_exactly_equal"]
    assert report["loader_parameters_equal_at_author_float32"]
    assert report["compared_parameter_arrays"] == 18
    field = report["cells"][0]["parameters"]["k"]
    assert field["reference_name"] == "asc_decay" and field["unit"] == "1/ms"
    assert field["raw_value"] == [0.3, 0.5]
    assert report["cells"][0]["reference_parameters_absent_from_loader"] == [
        "V_m", "adapting_threshold", "after_spike_currents", "asc_init", "spike_dependent_threshold"]
    assert not report["source_network_parameter_equality_verified"]
    assert not report["native_dynamics_qualified"]
    assert report["production_episodes"] == 0


def test_source_rounding_is_visible_and_not_a_fitted_tolerance():
    nodes = [node_group([0])]
    refs = [reference(1)]
    nodes[0]["params"]["g"] = np.float64(10.0) + 1e-8
    result = reconciliation.compare_parameters(nodes, np.array([1]), refs)
    assert not result["loader_parameters_exactly_equal"]
    assert result["loader_parameters_equal_at_author_float32"]
    assert result["cells"][0]["parameters"]["g"]["max_absolute_difference"] > 0
    nodes[0]["params"]["g"] = 10.01
    result = reconciliation.compare_parameters(nodes, np.array([1]), refs)
    assert result["status"] == "LOADER_PARAMETERS_REQUIRE_RECONCILIATION"
    assert not result["loader_parameters_equal_at_author_float32"]


@pytest.mark.parametrize("change", ["missing_reference", "duplicate_reference", "duplicate_node",
    "missing_node", "mixed_type", "split_type", "missing_parameter", "shape", "nonfinite", "wrong_family"])
def test_invalid_identity_or_parameters_fail(change):
    nodes = [node_group([0, 1]), node_group([2])]
    refs = [reference(10), reference(20)]
    if change == "missing_reference":
        refs.pop()
    elif change == "duplicate_reference":
        refs.append(deepcopy(refs[0]))
    elif change == "duplicate_node":
        nodes[1]["ids"] = np.array([0])
    elif change == "missing_node":
        nodes[0]["ids"] = np.array([0])
    elif change == "mixed_type":
        nodes = [node_group([0, 1, 2])]
    elif change == "split_type":
        nodes = [node_group([0]), node_group([1]), node_group([2])]
    elif change == "missing_parameter":
        del nodes[0]["params"]["t_ref"]
    elif change == "shape":
        nodes[0]["params"]["k"] = [0.1]
    elif change == "nonfinite":
        nodes[0]["params"]["g"] = np.inf
    else:
        refs[0]["converted_parameters"]["adapting_threshold"] = True
    with pytest.raises(ValueError, match="V1_"):
        reconciliation.compare_parameters(nodes, np.array([10, 10, 20]), refs)


def test_unknown_raw_parameters_cannot_be_silently_dropped():
    nodes = [node_group([0])]
    nodes[0]["params"]["unresolved_reset_rule"] = 1
    result = reconciliation.compare_parameters(nodes, np.array([1]), [reference(1)])
    assert result["loader_parameters_exactly_equal"]
    assert not result["all_raw_parameters_assessed"]
    assert result["status"] == "LOADER_PARAMETERS_REQUIRE_RECONCILIATION"
    assert result["cells"][0]["unassessed_raw_parameters"] == ["unresolved_reset_rule"]



def test_unassessed_numeric_values_are_disclosed_without_assessment():
    nodes = [node_group([0])]
    refs = [reference(1)]
    raw_initial = np.array([1.25, -2.5], dtype=np.float32)
    nodes[0]["params"]["asc_init"] = raw_initial
    refs[0]["converted_parameters"]["asc_init"] = np.array([0.0, 0.0], dtype=np.float64)
    refs[0]["converted_parameters"]["V_m"] = np.float32(-65.0)
    report = reconciliation.compare_parameters(nodes, np.array([1]), refs)
    row = report["cells"][0]
    assert row["unassessed_raw_parameters"] == ["asc_init"]
    assert row["unassessed_raw_parameter_values"] == {"asc_init": [1.25, -2.5]}
    disclosed = row["reference_parameter_values_absent_from_loader"]
    assert set(disclosed) == set(row["reference_parameters_absent_from_loader"])
    assert disclosed["asc_init"] == [0.0, 0.0]
    assert disclosed["V_m"] == -65.0
    assert report["loader_parameters_exactly_equal"]
    assert report["loader_parameters_equal_at_author_float32"]
    assert not report["all_raw_parameters_assessed"]
    assert report["status"] == "LOADER_PARAMETERS_REQUIRE_RECONCILIATION"
    assert report["production_episodes"] == 0
    assert not report["source_scientific_fidelity_qualified"]
    assert json.loads(json.dumps(report, allow_nan=False))["cells"][0] == row
    np.testing.assert_array_equal(nodes[0]["params"]["asc_init"], raw_initial)


@pytest.mark.parametrize("mismatch", [False, True])
def test_string_boolean_disclosure_keeps_unknown_and_mismatch_status(mismatch):
    nodes = [node_group([0])]
    refs = [reference(1)]
    nodes[0]["params"].update(V_dynamics_method=np.str_("source_method"),
                             raw_model_flag=np.bool_(True))
    refs[0]["converted_parameters"]["reference_method"] = np.str_("reference_method")
    if mismatch:
        nodes[0]["params"]["g"] = np.asarray(nodes[0]["params"]["g"]) + 1.0
    report = reconciliation.compare_parameters(nodes, np.array([1]), refs)
    row = report["cells"][0]
    assert row["unassessed_raw_parameters"] == ["V_dynamics_method", "raw_model_flag"]
    assert row["unassessed_raw_parameter_values"] == {"V_dynamics_method": "source_method", "raw_model_flag": True}
    disclosed = row["reference_parameter_values_absent_from_loader"]
    assert disclosed["reference_method"] == "reference_method"
    assert disclosed["after_spike_currents"] is True
    assert disclosed["spike_dependent_threshold"] is False
    assert disclosed["adapting_threshold"] is False
    assert report["loader_parameters_exactly_equal"] is (not mismatch)
    assert report["loader_parameters_equal_at_author_float32"] is (not mismatch)
    assert not report["all_raw_parameters_assessed"]
    assert report["status"] == "LOADER_PARAMETERS_REQUIRE_RECONCILIATION"
    assert not report["source_network_parameter_equality_verified"]
    assert not report["native_dynamics_qualified"]
    assert json.loads(json.dumps(report, allow_nan=False))["cells"][0] == row


def delivered(root, name, data, status):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    (root / "delivery.json").write_text(json.dumps(dict(status=status,
        files=[dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))])))


def archive_inputs(root, mismatch=False):
    inputs = archive_fixture(root)
    graph = inspect_archive(**inputs)
    graph_dir, reference_dir = root / "inspection", root / "reference"
    delivered(graph_dir, "graph_identity.json", graph, "INSPECTED_REQUIRES_RECONCILIATION")
    ref = reference(1)
    if mismatch:
        ref["converted_parameters"]["g"] += 1
    delivered(reference_dir, "reference/intake.json", dict(
        status="REFERENCE_PARAMETERS_ACQUIRED", bmtk_converter_sha256=BMTK_CONVERTER_SHA256,
        acquired_references=1, expected_specimens=1, references=[ref]), "REFERENCE_PARAMETERS_ACQUIRED")
    return dict(source_archive=inputs["archive"], graph_inspection_archive=graph_dir,
                parameter_reference_archive=reference_dir, output=root / "parameters.json")


@pytest.mark.parametrize("mismatch", [False, True])
def test_cli_preserves_comparison_and_returns_nonzero_on_difference(tmp_path, mismatch):
    inputs = archive_inputs(tmp_path, mismatch)
    argv = [item for key, path in inputs.items() for item in ("--" + key.replace("_", "-"), str(path))]
    assert reconciliation.main(argv) == (2 if mismatch else 0)
    result = json.loads(inputs["output"].read_text())
    assert result["cortical_neurons"] == 3
    assert result["loader_parameters_exactly_equal"] is not mismatch
    assert not result["source_network_parameter_equality_verified"]
    assert not result["source_scientific_fidelity_qualified"]


@pytest.mark.parametrize("change", ["reference_receipt", "graph_receipt", "archive_size", "graph_member_digest"])
def test_changed_consumed_input_is_rejected_before_pickle(tmp_path, change):
    inputs = archive_inputs(tmp_path)
    if change == "reference_receipt":
        (inputs["parameter_reference_archive"] / "reference/intake.json").write_text('{}')
    elif change == "graph_receipt":
        (inputs["graph_inspection_archive"] / "graph_identity.json").write_text('{}')
    elif change == "archive_size":
        with inputs["source_archive"].open("ab") as stream:
            stream.write(b"x")
    else:
        directory = inputs["graph_inspection_archive"]
        data = json.loads((directory / "graph_identity.json").read_text())
        data["member_sha256"][reconciliation.ARCHIVE_MEMBERS["graph"]] = "0" * 64
        delivered(directory, "graph_identity.json", data, "INSPECTED")
    with patch.object(reconciliation.pickle, "load") as load:
        with pytest.raises(ValueError, match="V1_"):
            reconciliation.reconcile_archive(**inputs)
        load.assert_not_called()

"""Bounded input-construction contracts; root runs the actual source canaries."""
import json
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np
import yaml

from . import v1_circuit_inputs as inputs
from . import v1_circuit_build as core
from .provenance import sha256_file


FIT_SOURCE = """
def get_filter_temporal_params(N, X, Y, model):
    basepath = 'build_files/biophys_props/lgn_fitted_models/'
    a = open_pickle(os.path.join(basepath, 'sOFF_TF4_fit.pkl'))
    b = open_pickle(os.path.join(basepath, 'tOFF_TF8_fit.pkl'))
    c = open_pickle(os.path.join(basepath, 'sON_TF4_fit.pkl'))
    for fname in os.listdir(basepath):
        pkl_savename = os.path.join(basepath, fname)
        if (tf_str in pkl_savename.split('_') and pkl_savename.find(cell_type) >= 0 and pkl_savename.find('.pkl') >= 0):
            filt_file = pkl_savename
"""


def reference_fixture(root):
    primary = root / "primary"; primary.mkdir(parents=True)
    for relative in inputs.REFERENCE_FILES:
        path = primary / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
    (primary / "report.json").write_text(json.dumps(dict(status="ORIGINAL_RECURRENT_CANARY_COMPLETE", mode="canary", exit_code=0)))
    rows = [dict(path="primary/" + name, bytes=(primary / name).stat().st_size, sha256=sha256_file(primary / name))
            for name in ("report.json", *inputs.REFERENCE_FILES)]
    (root / "delivery.json").write_text(json.dumps(dict(status="PASS_SOURCE_RECURRENT_CANARY", mode="canary", exit_code=0, files=rows)))


def nodes(directory, population, ids, kinds, types):
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory / f"{population}_nodes.h5", "w") as handle:
        group = handle.create_group("nodes/" + population)
        for key, values in (("node_id", ids), ("node_type_id", kinds), ("node_group_id", [0] * len(ids)), ("node_group_index", list(range(len(ids))))):
            group[key] = np.asarray(values, dtype=np.uint64)
        prop = group.create_group("0")
        for field in ("x", "y", "z", "tuning_angle"): prop[field] = np.arange(len(ids), dtype=float)
    (directory / f"{population}_node_types.csv").write_text(types)


def background_fixture(root):
    for kind in ("biophysical", "glif"):
        directory = root / kind
        nodes(directory, "v1", [10, 20], [1, 2] if kind == "biophysical" else [101, 102],
              "node_type_id pop_name\n1 eA\n2 eB\n" if kind == "biophysical" else "node_type_id pop_name\n101 eA\n102 eB\n")
        nodes(directory, "bkg", [0], [100], "node_type_id pop_name\n100 SG_001\n")
        with h5py.File(directory / "bkg_v1_edges.h5", "w") as handle:
            group = handle.create_group("edges/bkg_to_v1")
            for name, values in (("source_node_id", [0, 0]), ("target_node_id", [10, 20]),
                                 ("edge_type_id", [201, 202] if kind == "biophysical" else [301, 302]),
                                 ("edge_group_id", [0, 0]), ("edge_group_index", [0, 1])):
                group[name] = np.asarray(values, dtype=np.uint64)
            group.create_group("0")["nsyns"] = np.asarray([3, 4], dtype=np.uint32)
        table = "edge_type_id source_query target_query\n201 * node_type_id=='1'\n202 * node_type_id=='2'\n" if kind == "biophysical" else "edge_type_id source_query target_query\n301 * node_type_id=='101'\n302 * node_type_id=='102'\n"
        (directory / "bkg_v1_edge_types.csv").write_text(table)
    (root / "glif_props").mkdir()
    shutil.copy2(root / "glif/bkg_v1_edge_types.csv", root / "glif_props/bkg_v1_edge_types.csv")


class CircuitInputTests(unittest.TestCase):
    def test_reference_delivery_binding_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory(prefix="input-reference-") as temp:
            root = Path(temp); reference_fixture(root / "archive")
            primary, record = inputs.bind_reference(root / "archive", root / "output")
            self.assertEqual(primary["mode"], "canary")
            self.assertEqual(len(record["files"]), 5)
            self.assertFalse(record["recurrent_edges_read"])
            path = root / "archive/primary/glif/v1_nodes.h5"; path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "REFERENCE_DELIVERY_CHANGED"):
                inputs.bind_reference(root / "archive", root / "rejected")

    def test_numpy_checkpoint_and_python_stream_are_separate_and_replayable(self):
        np.random.seed(100); np.random.random(37); checkpoint = core.rng_state()
        expected = np.random.random(5)
        inputs.restore_numpy_state(checkpoint)
        random.seed(100); python_draws = [random.random() for _ in range(5)]
        np.testing.assert_array_equal(expected, np.random.random(5))
        random.seed(100); self.assertEqual(python_draws, [random.random() for _ in range(5)])
        with self.assertRaisesRegex(ValueError, "VALID_MT19937"):
            inputs.restore_numpy_state(dict(checkpoint, keys=checkpoint["keys"][:-1]))

    def test_original_fit_predicate_requires_missing_files_and_equal_duplicates(self):
        files = {name: dict(sha256=name) for name in ("sOFF_TF4_fit.pkl", "tOFF_TF8_fit.pkl", "sON_TF4_fit.pkl")}
        files.update({"sON_TF8_fit.pkl": dict(sha256="same"), "sON_TF8_fit (copy 1).pkl": dict(sha256="same")})
        selection = inputs.fit_selection(FIT_SOURCE, {"sON_TF8": {}}, files)
        self.assertEqual(len(selection["models"]["sON_TF8"]["matches"]), 2)
        files["sON_TF8_fit (copy 1).pkl"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_NONIDENTICAL_FITS"):
            inputs.fit_selection(FIT_SOURCE, {"sON_TF8": {}}, files)
        with self.assertRaisesRegex(ValueError, "MODEL_FIT_MISSING"):
            inputs.fit_selection(FIT_SOURCE, {"tOFF_TF15": {}}, files)
        del files["sON_TF4_fit.pkl"]
        with self.assertRaisesRegex(ValueError, "REQUIRED_FIXED_FIT_MISSING"):
            inputs.fit_selection(FIT_SOURCE, {"sON_TF8": {}}, files)

    def test_imported_cortical_identity_and_order_are_not_silently_changed(self):
        with tempfile.TemporaryDirectory(prefix="input-import-") as temp:
            root = Path(temp); background_fixture(root)
            class Node(dict):
                def __init__(self, identity, kind, population, value):
                    super().__init__(node_type_id=kind, pop_name=population, **{key: value for key in ("x", "y", "z", "tuning_angle")})
                    self.node_id = identity
            values = [Node(10, 1, "eA", 0.0), Node(20, 2, "eB", 1.0)]
            self.assertEqual(inputs.verify_imported_target(SimpleNamespace(nodes=lambda: values), root / "biophysical")["nodes"], 2)
            with self.assertRaisesRegex(ValueError, "IMPORTED_CORTICAL_NODE_ORDER_CHANGED"):
                inputs.verify_imported_target(SimpleNamespace(nodes=lambda: values[::-1]), root / "biophysical")

    def test_background_edge_identity_target_type_and_nsyns_are_verified(self):
        with tempfile.TemporaryDirectory(prefix="input-edges-") as temp:
            root = Path(temp); background_fixture(root)
            report = inputs.validate_inputs(root, "bkg", {}, 1)
            self.assertEqual((report["source_units"], report["edge_rows"], report["physical_synapses"]), (1, 2, 7))
            with h5py.File(root / "glif/bkg_v1_edges.h5", "a") as handle:
                handle["edges/bkg_to_v1/0/nsyns"][0] = 9
            with self.assertRaisesRegex(ValueError, "CONVERTED_NSYNS_CHANGED"):
                inputs.validate_inputs(root, "bkg", {}, 1)

    def test_mono_filter_nans_preserved_and_active_fields_must_be_finite(self):
        names = ["mono" + str(i) for i in range(12)] + sorted(inputs.DUAL_MODELS)
        models = {name: dict(N=8 if name not in inputs.DUAL_MODELS else 10) for name in names}
        ids = np.concatenate([np.full(models[name]["N"] * 150, i) for i, name in enumerate(names)])
        properties = {name: np.ones(17400) for name in ("x", "y", "spatial_size", *inputs.DOM_FIELDS, *inputs.OPTIONAL_FIELDS)}
        for name in inputs.OPTIONAL_FIELDS: properties[name][:14400] = np.nan
        data = dict(count=17400, values=dict(node_type_id=ids), types={i: dict(pop_name=name) for i, name in enumerate(names)}, properties=properties)
        self.assertEqual(inputs.validate_lgn_fields(data, models)["intentional_nan_values"], 14400 * 8)
        properties["tuning_angle"][0] = 0
        with self.assertRaisesRegex(ValueError, "MONO_FILTER_NAN_SEMANTICS_CHANGED"):
            inputs.validate_lgn_fields(data, models)
        properties["tuning_angle"][0] = np.nan; properties["weight_dom_0"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "NONFINITE_ACTIVE_LGN_FIELD"):
            inputs.validate_lgn_fields(data, models)

    def test_rng_mismatch_is_nonzero_while_native_comparison_evidence_is_retained(self):
        with tempfile.TemporaryDirectory(prefix="input-compare-") as temp:
            root = Path(temp)
            for label in ("a", "b"):
                work = root / label; background_fixture(work)
                for kind in ("biophysical", "glif"):
                    for suffix in ("nodes.h5", "node_types.csv", "v1_edges.h5", "v1_edge_types.csv"):
                        shutil.copy2(work / kind / ("bkg_" + suffix), work / kind / ("lgn_" + suffix))
                report = dict(status="ORIGINAL_CIRCUIT_INPUTS_COMPLETE", exit_code=0, mode="canary", rng=dict(test_state=label),
                    reference=dict(files={"bound": "same"}, delivery_sha256="same"), wheel=dict(sha256="same"),
                    fitted_models=dict(files={"same": {}}), sources={"same": dict(original_sha256="same")})
                (work / "report.json").write_text(json.dumps(report))
            result = inputs.compare_inputs(root / "a", root / "b", root / "comparison")
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["binding_differences"], ["rng"])
            self.assertEqual(len(result["files"]), 20)
            self.assertTrue(result["all_native_files_equal"])

    def test_bad_reference_leaves_failure_report(self):
        with tempfile.TemporaryDirectory(prefix="input-failure-") as temp:
            root = Path(temp); reference_fixture(root / "archive")
            path = root / "archive/delivery.json"; report = json.loads(path.read_text()); report["exit_code"] = 1; path.write_text(json.dumps(report))
            config = dict(source_root=str(root), bmtk_wheel=dict(path=str(root / "unused.whl"), sha256=core.WHEEL_SHA256),
                builder_seed=100, glif_assignment_seed=101, expected_full_nodes=230924, full_fraction=1.0, canary_fraction=0.01)
            path = root / "config.yaml"; path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "PASSED_RECURRENT_DELIVERY_REQUIRED"):
                inputs.build_inputs(path, root / "archive", root / "failed")
            self.assertEqual(json.loads((root / "failed/report.json").read_text())["exit_code"], 1)



class InputComputePlacementTests(unittest.TestCase):
    merlin_cpu_environment = {
        "RH2_JOB_RUN_ID": "fixture-merlin-run", "ARNOLD_TRIAL_ID": "fixture-merlin-trial",
        "ARNOLD_MONITOR_ROLE": "head", "ARNOLD_APPLIED_GPU_NUM": "0",
        "ARNOLD_HEAD_CPU": "2", "CUDA_VISIBLE_DEVICES": "",
    }

    def test_merlin_cpu_reaches_runtime_validation_and_records_actual_context(self):
        with tempfile.TemporaryDirectory(prefix="input-merlin-cpu-") as directory:
            root = Path(directory)
            with mock.patch.dict(inputs.os.environ, self.merlin_cpu_environment, clear=True), \
                 mock.patch.object(inputs.socket, "gethostname", return_value="merlin-fixture-host"), \
                 mock.patch.object(core, "load_config", return_value=({}, {})), \
                 mock.patch.object(inputs, "bind_reference", return_value=({"mode": "full"}, {})), \
                 mock.patch.object(inputs, "runtime_manifest", return_value={"fixture": True}), \
                 mock.patch.object(core, "verify_wheel", side_effect=RuntimeError("REACHED_RUNTIME_WHEEL_CHECK")) as wheel:
                with self.assertRaisesRegex(RuntimeError, "REACHED_RUNTIME_WHEEL_CHECK"):
                    inputs.build_inputs(root / "config.yaml", root / "recurrent", root / "output")
                wheel.assert_called_once()
                report = json.loads((root / "output/report.json").read_text())
                self.assertEqual(report["execution_context"], {
                    "backend": "merlin", "run_id": "fixture-merlin-run", "trial_id": "fixture-merlin-trial",
                    "hostname": "merlin-fixture-host", "allocated_cpus": 2, "allocated_gpus": 0,
                })
                self.assertNotIn("SLURMD_NODENAME", inputs.os.environ)
                self.assertNotIn("SLURM_JOB_ID", inputs.os.environ)

    def test_incomplete_merlin_identity_does_not_fall_back_to_legacy_node(self):
        cases = [("RH2_JOB_RUN_ID", None), ("ARNOLD_TRIAL_ID", None),
                 ("RH2_JOB_RUN_ID", ""), ("ARNOLD_TRIAL_ID", "")]
        for field, value in cases:
            environment = dict(self.merlin_cpu_environment, SLURMD_NODENAME="cpu03",
                               SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="1")
            if value is None:
                del environment[field]
            else:
                environment[field] = value
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch.dict(inputs.os.environ, environment, clear=True), \
                     mock.patch.object(core, "load_config", return_value=({}, {})), \
                     mock.patch.object(inputs, "bind_reference", return_value=({"mode": "full"}, {})), \
                     mock.patch.object(inputs, "runtime_manifest", return_value={"fixture": True}), \
                     mock.patch.object(core, "verify_wheel") as wheel:
                    with self.assertRaisesRegex(ValueError, "CPU_EXECUTION_INVALID_FIELD:" + field):
                        inputs.build_inputs(root / "config.yaml", root / "recurrent", root / "output")
                    wheel.assert_not_called()
                    report = json.loads((root / "output/report.json").read_text())
                    self.assertEqual(report["exit_code"], 1)
                    self.assertNotIn("execution_context", report)

    def test_merlin_gpu_allocation_or_visible_device_is_rejected(self):
        for field, value in (("ARNOLD_APPLIED_GPU_NUM", "1"), ("CUDA_VISIBLE_DEVICES", "0")):
            environment = dict(self.merlin_cpu_environment, SLURMD_NODENAME="cpu03",
                               SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="1")
            environment[field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch.dict(inputs.os.environ, environment, clear=True), \
                     mock.patch.object(core, "load_config", return_value=({}, {})), \
                     mock.patch.object(inputs, "bind_reference", return_value=({"mode": "full"}, {})), \
                     mock.patch.object(inputs, "runtime_manifest", return_value={"fixture": True}), \
                     mock.patch.object(core, "verify_wheel") as wheel:
                    with self.assertRaisesRegex(ValueError, "CPU_EXECUTION_INVALID_FIELD:" + field):
                        inputs.build_inputs(root / "config.yaml", root / "recurrent", root / "output")
                    wheel.assert_not_called()
                    report = json.loads((root / "output/report.json").read_text())
                    self.assertEqual(report["exit_code"], 1)
                    self.assertNotIn("execution_context", report)

    def test_supported_nodes_reach_runtime_validation(self):
        cases = [
            ("full", "cpu03", True), ("full", "cpu05", True), ("full", "cpu06", True),
            ("full", "cpu04", False), ("canary", "cpu04", True), ("canary", "cpu05", True),
            ("full", "mgmt2", False),
        ]
        for mode, node, allowed in cases:
            with self.subTest(mode=mode, node=node), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch.dict(inputs.os.environ, {"SLURMD_NODENAME": node}, clear=True), \
                     mock.patch.object(core, "load_config", return_value=({}, {})), \
                     mock.patch.object(inputs, "bind_reference", return_value=({"mode": mode}, {})), \
                     mock.patch.object(inputs, "runtime_manifest", return_value={"fixture": True}), \
                     mock.patch.object(core, "verify_wheel", side_effect=RuntimeError("REACHED_RUNTIME_WHEEL_CHECK")) as wheel:
                    error = RuntimeError if allowed else ValueError
                    message = "REACHED_RUNTIME_WHEEL_CHECK" if allowed else "V1_INPUT_COMPUTE_PLACEMENT_INVALID"
                    with self.assertRaisesRegex(error, message):
                        inputs.build_inputs(root / "config.yaml", root / "recurrent", root / "output")
                    self.assertEqual(wheel.called, allowed)

if __name__ == "__main__":
    unittest.main()

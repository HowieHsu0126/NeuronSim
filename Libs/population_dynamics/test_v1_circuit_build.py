"""Small fixtures for scientific contracts; source canaries run in separate jobs."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import yaml

from . import v1_circuit_build as build
from .provenance import sha256_file


def config_fields():
    return dict(source_root="sources", bmtk_wheel=dict(path="bmtk.whl", sha256=build.WHEEL_SHA256),
                builder_seed=100, glif_assignment_seed=101, expected_full_nodes=230924,
                canary_fraction=0.01, full_fraction=1.0, edge_chunk_size=2)


def native_fixture(root):
    mapping = {"1": dict(node_type_id=[101, 102], pop_name="eA", proportions=[0.5, 0.5])}
    for label in ("biophysical", "glif"):
        directory = root / label; directory.mkdir(parents=True)
        with h5py.File(directory / "v1_nodes.h5", "w") as handle:
            group = handle.create_group("nodes/v1")
            group["node_id"] = np.asarray([10, 30, 20], dtype=np.uint64)
            group["node_type_id"] = np.asarray([1, 1, 1] if label == "biophysical" else [101, 102, 101], dtype=np.uint64)
            group["node_group_id"] = np.zeros(3, dtype=np.uint32)
            group["node_group_index"] = np.arange(3, dtype=np.uint64)
            group.create_group("0")["x"] = np.asarray([0.0, 1.0, 2.0])
        if label == "biophysical":
            nodes = "node_type_id model_type pop_name\n1 biophysical eA\n"
            edges = "edge_type_id source_query target_query\n1 pop_name=='eA' node_type_id=='1'\n2 pop_name=='eA' node_type_id=='1'\n"
        else:
            nodes = "node_type_id model_type pop_name\n101 point_process eA\n102 point_process eA\n"
            edges = "edge_type_id source_query target_query\n11 pop_name=='eA' node_type_id=='101'\n12 pop_name=='eA' node_type_id=='102'\n"
        (directory / "v1_node_types.csv").write_text(nodes)
        (directory / "v1_v1_edge_types.csv").write_text(edges)
        with h5py.File(directory / "v1_v1_edges.h5", "w") as handle:
            group = handle.create_group("edges/v1_to_v1")
            group["source_node_id"] = np.asarray([10, 20, 30, 10], dtype=np.uint64)
            group["target_node_id"] = np.asarray([20, 30, 10, 30], dtype=np.uint64)
            group["edge_type_id"] = np.asarray([1, 2, 1, 2] if label == "biophysical" else [11, 12, 11, 12], dtype=np.int64)
            group["edge_group_id"] = np.zeros(4, dtype=np.uint32)
            group["edge_group_index"] = np.arange(4, dtype=np.uint64)
            group.create_group("0")["nsyns"] = np.asarray([2, 3, 4, 5], dtype=np.uint32)
    (root / "glif_props").mkdir()
    (root / "glif_props/v1_v1_edge_types.csv").write_text((root / "glif/v1_v1_edge_types.csv").read_text())
    report = dict(status="ORIGINAL_RECURRENT_CANARY_COMPLETE", exit_code=0, mode="canary",
                  resolved_config=config_fields(), wheel=dict(sha256=build.WHEEL_SHA256),
                  sources={"fixture": dict(original_sha256="fixture-only")})
    (root / "report.json").write_text(json.dumps(report))
    return mapping


class CircuitBuildTests(unittest.TestCase):
    def test_config_freezes_full_scale_and_separate_seeds(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-config-") as temp:
            path = Path(temp) / "config.yaml"
            fields = config_fields(); path.write_text(yaml.safe_dump(fields))
            config, snapshots = build.load_config(path)
            self.assertEqual(config["builder_seed"], 100)
            self.assertEqual(config["glif_assignment_seed"], 101)
            self.assertEqual(snapshots["original_config"], fields)
            fields["full_fraction"] = 0.1; path.write_text(yaml.safe_dump(fields))
            with self.assertRaisesRegex(ValueError, "FROZEN_CONFIG_DIFFERS:full_fraction"):
                build.load_config(path)

    def test_config_accepts_independent_circuit_seeds_and_rejects_invalid_streams(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-seeds-") as temp:
            path = Path(temp) / "config.yaml"
            fields = config_fields()
            fields.update(builder_seed=100000000, glif_assignment_seed=101000000)
            path.write_text(yaml.safe_dump(fields))
            config, _ = build.load_config(path)
            self.assertEqual(config["builder_seed"], 100000000)
            self.assertEqual(config["glif_assignment_seed"], 101000000)
            first = build.seed_and_record(config["builder_seed"])
            self.assertNotEqual(first, build.seed_and_record(config["builder_seed"] + 1))
            for value, error in ((True, "INTEGER_CONFIG"), (-1, "UINT32_SEED"),
                                 (2**32, "UINT32_SEED"), (1.5, "INTEGER_CONFIG"),
                                 (101000000, "SEPARATE_RANDOM_STREAM")):
                fields["builder_seed"] = value
                path.write_text(yaml.safe_dump(fields))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, error):
                    build.load_config(path)
            del fields["builder_seed"]
            path.write_text(yaml.safe_dump(fields))
            with self.assertRaisesRegex(ValueError, "INTEGER_CONFIG"):
                build.load_config(path)

    def test_source_receipts_and_only_declared_alias_changes(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-staging-") as temp:
            root = Path(temp); source = root / "source"; output = root / "work"; output.mkdir()
            for role in build.SOURCE_FILES:
                directory = source / role; directory.mkdir(parents=True)
                text = "x=np.NaN\ny=np.NaN\n" if role == "original_edge_functions" else "dtype=np.int\nnodes_df.loc[subgrp.index.values, 'glif_node_type_id'] = glif_ntids\n" if role.endswith("legacy_glif_converter") else "{}"
                path = directory / "source_asset.txt"; path.write_text(text)
                (directory / "source.json").write_text(json.dumps(dict(artifact=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))))
            bodies, records = build.stage_sources(source, output)
            self.assertEqual(len(records), 11)
            self.assertEqual(len(records["original_edge_functions"]["compatibility"]), 2)
            self.assertEqual(len(records["weight_semantics/legacy_glif_converter"]["compatibility"]), 2)
            self.assertIn("np.NaN", bodies["original_edge_functions"].decode())
            self.assertNotIn("np.NaN", (output / "build_files/edge_funcs.py").read_text())
            assignment = (output / "convert_bio2glif.py").read_text().splitlines()[1]
            frame = pd.DataFrame({"glif_node_type_id": [0, 0, 0], "label": ["x", "x", "y"]})
            namespace = dict(nodes_df=frame, subgrp=frame.iloc[:2], glif_ntids=[101])
            exec(assignment, namespace)
            self.assertEqual(frame["glif_node_type_id"].tolist(), [101, 101, 0])
            (source / "original_build_network/source_asset.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SOURCE_RECEIPT_MISMATCH"):
                build.stage_sources(source, root / "failed-work")

    def test_estimate_keeps_source_order_lif_prefix_and_exact_dense_count(self):
        document = dict(locations={"L": {"eA": dict(models=[dict(node_type_id=1, N=200000, model_type="biophysical")]),
                                         "iB": dict(models=[dict(node_type_id=2, N=30924, model_type="point_process")])}})
        with tempfile.TemporaryDirectory(prefix="v1-build-estimate-") as temp:
            work = Path(temp); path = work / build.SOURCE_FILES["original_edge_parameters"]
            path.parent.mkdir(parents=True)
            path.write_text("edge_type_id source_label target_label target_model_id\n5 eA LIFiB 2\n6 eA eA 1\n7 LIFiB eA 1\n")
            calls = []
            def parameters(source, target):
                calls.append((source, target))
                return dict(A_new=0.0 if (source, target) == ("eA", "eA") else 0.5)
            result = build.estimate_dense({"original_node_parameters": json.dumps(document).encode()}, work, 0.01, parameters)
            self.assertEqual(result["nodes"], 2309)
            self.assertEqual(result["candidate_pairs"], 2 * 2000 * 309)
            self.assertEqual(result["dense_nsyns_bytes"], 4 * 2 * 2000 * 309)
            self.assertEqual([row["csv_row"] for row in result["maps"]], [0, 2])
            self.assertEqual(calls, [("eA", "LIFiB"), ("eA", "eA"), ("LIFiB", "eA")])

    def test_rng_full_count_memory_and_component_guards(self):
        first = build.seed_and_record(100); draws = np.random.randint(0, 100000, 8)
        self.assertNotEqual(first, build.rng_state())
        self.assertEqual(first, build.seed_and_record(100))
        np.testing.assert_array_equal(draws, np.random.randint(0, 100000, 8))
        self.assertNotEqual(first, build.seed_and_record(101))
        build.require_node_count("canary", 3, 3)
        with self.assertRaisesRegex(ValueError, "NODE_COUNT_DIFFERS"):
            build.require_node_count("full", 3, 3)
        with self.assertRaisesRegex(ValueError, "MEMORY_BELOW_DENSE_LOWER_BOUND"):
            build.require_memory_lower_bound("full", 5 * 1024**3, {"SLURM_MEM_PER_NODE": "4096"})
        record = build.require_memory_lower_bound("full", 1024**3, {"SLURM_MEM_PER_NODE": "4096"})
        self.assertFalse(record["peak_memory_qualified"])
        build.require_coupled_component("canary", dict(largest_strong_component=3))
        with self.assertRaisesRegex(ValueError, "COMPONENT_BELOW_100000"):
            build.require_coupled_component("full", dict(largest_strong_component=99999))

    def test_native_validation_counts_synapses_separately_and_allows_stochastic_type_counts(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-native-") as temp:
            root = Path(temp); mapping = native_fixture(root)
            report = build.validate_native(root, mapping, "canary", 3, 2, expected_synapses=14)
            self.assertEqual(report["edge_rows"], 4)
            self.assertEqual(report["physical_synapses"], 14)
            self.assertEqual(report["all_edge_rows_verified"], 4)
            self.assertEqual(report["glif_type_counts"], {"101": 2, "102": 1})
            self.assertEqual(report["connectivity"]["largest_strong_component"], 3)
            self.assertFalse(report["full_scale_gate_applied"])
            records = build.table_records(root / "biophysical/v1_v1_edge_types.csv", "edge_type_id")
            with self.assertRaisesRegex(ValueError, "EDGE_TABLE_PAIR_INVALID"):
                build.edge_type_arrays(records, {"eA": 0}, require_unique_pairs=True)
            self.assertEqual(build.edge_type_arrays(records, {"eA": 0}, require_unique_pairs=False)[0].tolist(), [1, 2])

    def test_native_wrong_edge_type_nsyns_and_unknown_node_fail(self):
        for mutation, error in (("type", "EDGE_TYPE_PREDICATE_DIFFERS"), ("nsyns", "CONVERTED_NSYNS_CHANGED"), ("node", "EDGE_REFERENCES_UNKNOWN_NODE")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="v1-build-invalid-") as temp:
                root = Path(temp); mapping = native_fixture(root)
                if mutation == "node":
                    for network in ("biophysical", "glif"):
                        with h5py.File(root / network / "v1_v1_edges.h5", "a") as handle:
                            handle["edges/v1_to_v1/source_node_id"][0] = 999
                else:
                    with h5py.File(root / "glif/v1_v1_edges.h5", "a") as handle:
                        handle["edges/v1_to_v1/edge_type_id" if mutation == "type" else "edges/v1_to_v1/0/nsyns"][0] = 12 if mutation == "type" else 9
                with self.assertRaisesRegex(ValueError, error):
                    build.validate_native(root, mapping, "canary", 3, 2)

    def test_every_native_dataset_and_semantic_table_are_compared_and_mismatch_is_nonzero(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-compare-") as temp:
            root = Path(temp); left, right = root / "left", root / "right"
            native_fixture(left); native_fixture(right)
            path = right / "glif/v1_node_types.csv"
            frame = pd.read_csv(path, sep=" "); frame.iloc[::-1][list(reversed(frame.columns))].to_csv(path, sep=" ", index=False)
            report = build.compare_builds(left, right, root / "equal", chunk_size=2)
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual(len(report["files"]), 8)
            with h5py.File(right / "glif/v1_nodes.h5", "a") as handle:
                handle["nodes/v1/0/x"][2] = 77.0
            code = build.main(["compare", "--left", str(left), "--right", str(right), "--output", str(root / "different"), "--chunk-size", "2"])
            self.assertEqual(code, 1)
            recorded = json.loads((root / "different/report.json").read_text())
            self.assertEqual(recorded["status"], "ORIGINAL_RECURRENT_REPLAY_DIFFERS")
            changed = next(row for row in recorded["files"] if row["path"] == "glif/v1_nodes.h5")
            self.assertEqual(changed["differences"][0]["dataset"], "nodes/v1/0/x")

    def test_bad_wheel_is_rejected_and_failure_report_is_retained(self):
        with tempfile.TemporaryDirectory(prefix="v1-build-wheel-") as temp:
            root = Path(temp); wheel = root / "bad.whl"; wheel.write_bytes(b"not a wheel")
            fields = config_fields(); fields["bmtk_wheel"]["path"] = str(wheel)
            config = root / "config.yaml"; config.write_text(yaml.safe_dump(fields))
            with self.assertRaisesRegex(ValueError, "WHEEL_CONTENT_DIFFERS"):
                build.execute(config, "canary", root / "failed")
            record = json.loads((root / "failed/report.json").read_text())
            self.assertEqual(record["exit_code"], 1)
            self.assertFalse(record["full_recurrent_realization"])


if __name__ == "__main__":
    unittest.main()

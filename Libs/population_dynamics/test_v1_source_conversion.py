"""Exercise the preserved converter, real entry point, and provenance failures."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from . import v1_source_conversion as conversion
from .provenance import sha256_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("NRNSIM_CONVERTER_SOURCE", str(ROOT / "Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/weight_semantics/legacy_glif_converter/source_asset.txt")))
TABLE = """edge_type_id source_query target_query
1 pop_name=='eA' node_type_id=='101'
2 pop_name=='eA' node_type_id=='102'
3 pop_name=='eB' node_type_id=='101'
4 pop_name=='eB' node_type_id=='102'
"""


def fixture(root):
    nodes, types = root / "nodes.h5", root / "types.csv"
    with h5py.File(nodes, "w") as handle:
        group = handle.create_group("nodes/v1")
        group["node_id"] = np.asarray([70, 20, 10, 90], dtype=np.uint64)
        group["node_type_id"] = np.asarray([101, 102, 101, 102], dtype=np.uint64)
    types.write_text("node_type_id pop_name\n101 eA\n102 eB\n")
    source_root = root / "sources"
    source_hashes = {}
    for name, body in (("legacy_glif_converter", SOURCE.read_text()), ("legacy_recurrent_table", TABLE)):
        directory = source_root / "weight_semantics" / name
        directory.mkdir(parents=True)
        path = directory / "source_asset.txt"; path.write_text(body)
        receipt = directory / "source.json"
        entry = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
        receipt.write_text(json.dumps(dict(entry, artifact=path.name)))
        source_hashes["weight_semantics/" + name] = dict(entry, receipt_sha256=sha256_file(receipt))
    cases = [dict(name="case" + str(i), source_id=s, target_id=t, raw_group=i, raw_offset=0)
             for i, (s, t) in enumerate(((70, 90), (20, 10), (10, 20)))]
    arithmetic = root / "arithmetic.json"
    arithmetic.write_text(json.dumps(dict(status="SOURCE_WEIGHT_ARITHMETIC_DIAGNOSTIC_COMPLETE", cases=cases,
        input_sha256=dict(nodes=sha256_file(nodes), types=sha256_file(types)),
        source_sha256=source_hashes, comparison_report_sha256="bound-comparison")))
    sonata_cases = [dict(case, matches_complete=True, matching_row_count=1, matching_rows=[dict(
        source_node_id=case["source_id"], target_node_id=case["target_id"], edge_type_id=99,
        global_row_id=100 + i, properties=dict(nsyns=dict(status="RECORDED", value=i + 2)))])
        for i, case in enumerate(cases)]
    sonata = root / "sonata.json"
    sonata.write_text(json.dumps(dict(status="SOURCE_SONATA_WITNESS_QUERY_COMPLETE", completed_selectors=3,
        cases=sonata_cases, witness_report_sha256=sha256_file(arithmetic), comparison_report_sha256="bound-comparison")))
    comparison = root / "node_comparison.json"
    comparison.write_text(json.dumps(dict(status="COMPLETE_NODE_METADATA_COMPARISON", comparison=dict(nodes=4),
        legacy_asset=dict(complete=True, bytes=nodes.stat().st_size, sha256=sha256_file(nodes)),
        input_sha256={"current_nodes.h5": sha256_file(nodes), "current_types.txt": sha256_file(types),
                      "arithmetic.json": sha256_file(arithmetic), "sonata.json": sha256_file(sonata)})))
    return dict(source_root=source_root, nodes=nodes, types=types, node_comparison=comparison,
                arithmetic_report=arithmetic, sonata_report=sonata, output=root / "out")


class SourceConversionTests(unittest.TestCase):
    def test_original_function_cli_preserves_rows_and_reports_source_disagreement(self):
        with tempfile.TemporaryDirectory(prefix="source-conversion-test-") as temp:
            args = fixture(Path(temp))
            argv = [item for key, value in args.items() for item in ("--" + key.replace("_", "-"), str(value))]
            self.assertEqual(conversion.main(argv), 0)
            report = json.loads((args["output"] / "report.json").read_text())
            self.assertEqual(report["status"], "SOURCE_CONVERTER_REPLAY_DIFFERS_FROM_PUBLISHED_WITNESSES")
            self.assertEqual(report["unique_witnesses"], 3)
            self.assertEqual(report["result"]["independent_lookup_matches"], 9)
            self.assertEqual(report["result"]["published_source_matches"], 0)
            self.assertTrue(all(report["result"]["preserved_fields"].values()))
            self.assertEqual(report["result"]["block_intervals"], [[0, 4], [4, 8], [8, 9]])
            self.assertFalse(report["scientific_qualification"])
            self.assertFalse(report["historical_producer_identified"])
            self.assertEqual(len(report["function"]["compatibility"]), 1)
            self.assertEqual([row["generated_edge_type_id"] for row in report["result"]["rows"]],
                             [2, 2, 3, 3, 2, 2, 2, 2, 3])

    def test_positional_miswrite_returns_nonzero_and_preserves_diagnostic(self):
        with tempfile.TemporaryDirectory(prefix="source-conversion-miswrite-") as temp:
            args = fixture(Path(temp))
            original, function_report = conversion.extract_converter(SOURCE.read_text())

            def miswrite(*positional, **keywords):
                original(*positional, **keywords)
                path = Path(keywords["glif_network_dir"]) / "v1_v1_edges.h5"
                with h5py.File(path, "a") as handle:
                    handle["edges/v1_to_v1/edge_type_id"][0] = 99

            argv = [item for key, value in args.items()
                    for item in ("--" + key.replace("_", "-"), str(value))]
            with mock.patch.object(conversion, "extract_converter", return_value=(miswrite, function_report)):
                self.assertEqual(conversion.main(argv), 2)
            report = json.loads((args["output"] / "report.json").read_text())
            self.assertEqual(report["status"], "SOURCE_CONVERTER_REPLAY_POSITIONAL_MISMATCH")
            self.assertEqual(report["result"]["independent_lookup_matches"], 8)
            self.assertTrue((args["output"] / "slice/diagnostic_converted_edges.h5").is_file())

    def test_ast_extraction_does_not_run_module_driver_or_other_top_level_code(self):
        function, report = conversion.extract_converter(SOURCE.read_text() + "\nraise RuntimeError('must not execute')\n")
        self.assertEqual(function.__name__, "convert_edges")
        self.assertIn("def convert_edges", report["original_function_text"])
        self.assertEqual(report["compatibility"][0]["before"], "np.int")

    def test_changed_node_input_fails_before_converter_execution_and_records_failure(self):
        with tempfile.TemporaryDirectory(prefix="source-conversion-binding-") as temp:
            args = fixture(Path(temp))
            with h5py.File(args["nodes"], "a") as handle:
                handle["nodes/v1/node_type_id"][0] = 102
            with mock.patch.object(conversion, "extract_converter") as extract:
                with self.assertRaisesRegex(ValueError, "BOUND_METADATA_OR_WITNESS_CHANGED"):
                    conversion.replay(**args)
                extract.assert_not_called()
            report = json.loads((args["output"] / "report.json").read_text())
            self.assertEqual(report["exit_code"], 1)
            self.assertEqual(report["status"], "FAILED_SOURCE_CONVERTER_REPLAY")

    def test_source_receipt_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="source-conversion-source-") as temp:
            args = fixture(Path(temp))
            directory = args["source_root"] / "weight_semantics/legacy_glif_converter"
            with (directory / "source_asset.txt").open("a") as handle:
                handle.write("\n# changed\n")
            with self.assertRaisesRegex(ValueError, "BOUND_SOURCE_CHANGED"):
                conversion.replay(**args)

    def test_duplicate_node_ids_and_missing_witness_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="source-conversion-identities-") as temp:
            args = fixture(Path(temp))
            with h5py.File(args["nodes"], "a") as handle:
                handle["nodes/v1/node_id"][0] = 20
            with self.assertRaisesRegex(ValueError, "UNIQUE_NATIVE_IDS_REQUIRED"):
                conversion.node_dataframe(args["nodes"], args["types"])
            with h5py.File(args["nodes"], "a") as handle:
                handle["nodes/v1/node_id"][0] = 70
            with self.assertRaisesRegex(ValueError, "WITNESS_NODE_ID_MISSING"):
                conversion.replay_slice(None, args["nodes"], args["types"], TABLE,
                    [dict(source_node_id=999, target_node_id=10)], args["output"])

    def test_duplicate_lookup_keys_and_incomplete_source_witness_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNIQUE_EDGE_TYPE_KEYS_REQUIRED"):
            conversion.edge_lookup(TABLE + "5 pop_name=='eA' node_type_id=='101'\n")
        with tempfile.TemporaryDirectory(prefix="source-conversion-witness-") as temp:
            args = fixture(Path(temp))
            arithmetic = json.loads(args["arithmetic_report"].read_text())
            sonata = json.loads(args["sonata_report"].read_text())
            sonata["cases"][0]["matches_complete"] = False
            with self.assertRaisesRegex(ValueError, "BOUND_WITNESS_IDENTITY_REQUIRED"):
                conversion.witness_rows(arithmetic, sonata)


if __name__ == "__main__":
    unittest.main()

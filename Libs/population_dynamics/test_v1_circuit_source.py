from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from population_dynamics.v1_circuit_source import read_source, declared_models, check_mapping, edge_pairs


class CircuitSourceTests(unittest.TestCase):
    def setUp(self):
        self.bio = {11: {"pop_name": "e23"}}
        self.glif = {21: {"pop_name": "e23"}, 22: {"pop_name": "e23"}}
        self.mapping = {"11": {"node_type_id": [21, 22], "proportions": [0.25, 0.75], "pop_name": "e23"}}

    def test_source_receipt_binds_exact_payload(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name);raw=b"source bytes"
            (path / "source_asset.txt").write_bytes(raw)
            (path / "source.json").write_text(json.dumps({"artifact": "source_asset.txt", "bytes": len(raw),
                                                          "sha256": hashlib.sha256(raw).hexdigest()}))
            self.assertEqual(read_source(path)[0], raw)
            (path / "source_asset.txt").write_bytes(b"source Bytes")
            with self.assertRaisesRegex(ValueError, "RECEIPT_MISMATCH"):
                read_source(path)

    def test_mapping_preserves_full_model_and_population_coverage(self):
        result = check_mapping(self.bio, self.glif, self.mapping)
        self.assertEqual(result["covered_glif_types"], 2)
        with self.assertRaisesRegex(ValueError, "MISSING_SOURCE_TYPES"):
            check_mapping(self.bio, self.glif, {})
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_GLIF_COVERAGE"):
            check_mapping(self.bio, self.glif, {"11": {"node_type_id": [21], "pop_name": "e23"}})
        with self.assertRaisesRegex(ValueError, "POPULATION_DIFFERS"):
            check_mapping(self.bio, {21: {"pop_name": "e23"}, 22: {"pop_name": "i23"}}, self.mapping)

    def test_invalid_categorical_probabilities_are_rejected(self):
        for weights in ([0.25, 0.25], [-0.25, 1.25], [float("nan"), 0.75], [1.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                check_mapping(self.bio, self.glif, {"11": dict(self.mapping["11"], proportions=weights)})

    def test_edge_queries_are_parsed_without_evaluation(self):
        header = "target_query source_query syn_weight delay\n"
        row = "node_type_id=='21' pop_name=='e23' -0.0 1.5\n"
        self.assertEqual(edge_pairs(header + row, self.glif, {"e23": {}}), {(21, "e23")})
        for invalid in (row + row, row.replace("'21'", "'23'"), row.replace("1.5", "nan"),
                        row.replace("node_type_id=='21'", "__import__('os')")):
            with self.subTest(row=invalid), self.assertRaises(ValueError):
                edge_pairs(header + invalid, self.glif, {"e23": {}})

    def test_duplicate_model_identity_is_not_overwritten(self):
        document = {"locations": {"L23": {"e23": {"models": [{"node_type_id": 21, "N": 2},
                                                                  {"node_type_id": 21, "N": 3}]}}}}
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_INVALID_MODEL"):
            declared_models(document)


if __name__ == "__main__":
    unittest.main()

"""Synthetic engineering checks; these fixtures are not native-source evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

import numpy as np

from . import pfc_stream_comparison as comparison
from . import test_pfc_source_scale_analysis as fixtures
from .provenance import sha256_file


class StreamComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceReadoutTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)

    def pair(self, suffix=""):
        candidate = self.fixture.archive("pfc_94001_strong_cue", suffix="_candidate" + suffix)
        reference = self.fixture.archive("pfc_94001_strong_cue", suffix="_reference" + suffix)
        return candidate, reference

    def run_comparison(self, candidate, reference, name="comparison", **kwargs):
        return comparison.compare(candidate, reference, self.fixture.config_path,
                                  self.fixture.root / name, **kwargs)

    def test_identical_legacy_arrays_do_not_claim_external_stream_or_current_equality(self):
        report = self.run_comparison(*self.pair())
        self.assertEqual(report["status"], "COMPLETE_RECORDED_ARRAYS_IDENTICAL")
        self.assertTrue(all(row["exact_equal"] for row in report["array_results"]))
        self.assertEqual({row["artifact"] for row in report["array_results"]}, set(comparison.ARTIFACTS))
        self.assertEqual(report["current_observations"]["comparison_status"], "UNAVAILABLE")
        self.assertEqual(report["current_observations"]["reference"]["availability"], "UNAVAILABLE")
        self.assertFalse(report["full_external_streams_compared"])
        self.assertFalse(report["external_input_equivalence_established"])
        self.assertFalse(report["scientific_fidelity_qualified"])
        self.assertEqual(report["new_episodes"], 0)
        self.assertEqual(json.loads((self.fixture.root / "comparison/report.json").read_text()), report)

    def test_all_recorded_differences_are_retained_as_completed_diagnostic(self):
        candidate, reference = self.pair()
        self.fixture.modify_array(candidate, "terminal_state.npz", "E__s_AMPA",
                                  lambda values: values.__setitem__(3, .125))
        self.fixture.modify_array(candidate, "final_rng.npz", "keys",
                                  lambda values: values.__setitem__(1, 9))
        initial = np.load(candidate / "initial_voltage.npy", allow_pickle=False)
        initial[0] += 1
        np.save(candidate / "initial_voltage.npy", initial)
        self.fixture.rebind(candidate)
        report = self.run_comparison(candidate, reference)
        self.assertEqual(report["status"], "COMPLETE_RECORDED_ARRAYS_DIFFERENT")
        different = {(row["artifact"], row["array"]): row for row in report["array_results"]
                     if not row["exact_equal"]}
        self.assertEqual(set(different), {
            ("terminal_state.npz", "E__s_AMPA"), ("final_rng.npz", "keys"),
            ("initial_voltage.npy", "initial_voltage_mV")})
        self.assertEqual(different["terminal_state.npz", "E__s_AMPA"]["value_mismatch_count"], 1)
        self.assertEqual(different["final_rng.npz", "keys"]["max_abs_numeric_delta"], 7)
        self.assertTrue(report["comparison_completed"])

    def test_scalar_shape_dtype_and_large_integer_differences(self):
        scalar = comparison._compare_array("rng", "position", np.array(5), np.array(4))
        self.assertEqual(scalar["max_abs_numeric_delta"], 1)
        dtype = comparison._compare_array("rng", "keys", np.array([4], dtype=np.uint64),
                                          np.array([4], dtype=np.uint32))
        self.assertFalse(dtype["exact_equal"])
        self.assertEqual(dtype["value_mismatch_count"], 0)
        shape = comparison._compare_array("raw", "ticks", np.array([1, 2]), np.array([1]))
        self.assertEqual(shape["status"], "SHAPE_MISMATCH")
        self.assertIsNone(shape["value_mismatch_count"])
        large = comparison._compare_array("rng", "keys",
            np.array([2**64 - 1], dtype=np.uint64), np.array([2**64 - 2], dtype=np.uint64))
        self.assertEqual(large["max_abs_numeric_delta"], 1)

    def test_one_sided_and_both_recorded_current_availability(self):
        candidate, reference = self.pair()
        self.fixture.add_current(candidate)
        one = self.run_comparison(candidate, reference, "one_current")
        self.assertEqual(one["status"], "COMPLETE_RECORDED_ARRAYS_IDENTICAL")
        self.assertEqual(one["current_observations"]["candidate"]["availability"], "AVAILABLE")
        self.assertEqual(one["current_observations"]["comparison_status"], "UNAVAILABLE")
        self.assertFalse(any(row["artifact"] == comparison.CURRENT_ARTIFACT for row in one["array_results"]))
        self.fixture.add_current(reference)
        both = self.run_comparison(candidate, reference, "both_current")
        self.assertEqual(both["current_observations"]["comparison_status"], "COMPARED")
        self.assertTrue(any(row["artifact"] == comparison.CURRENT_ARTIFACT for row in both["array_results"]))

    def test_different_case_and_changed_configuration_raise(self):
        candidate, reference = self.pair()
        other = self.fixture.archive("pfc_94002_strong_cue")
        with self.assertRaisesRegex(ValueError, "CASE_MISMATCH"):
            self.run_comparison(candidate, other, "wrong_case")
        changed = dict(self.fixture.config, distractor_center_deg=91.)
        config = self.fixture.root / "changed.yaml"
        config.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            comparison.compare(candidate, reference, config, self.fixture.root / "wrong_config")
        self.assertEqual(json.loads((self.fixture.root / "wrong_config/report.json").read_text())["status"], "FAILED")

    def test_corrupted_receipt_binding_and_nonfinite_rng_raise(self):
        candidate, reference = self.pair()
        with (candidate / "initial_voltage.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_comparison(candidate, reference, "corrupt")
        candidate, reference = self.pair("_nan")
        np.savez(candidate / "final_rng.npz", keys=np.array([1], dtype=np.uint32),
                 position=np.array(float("nan")))
        self.fixture.rebind(candidate)
        with self.assertRaisesRegex(ValueError, "NONFINITE_ARRAY"):
            self.run_comparison(candidate, reference, "nonfinite")

    def test_backend_versions_and_producer_code_are_independently_bound(self):
        candidate, reference = self.pair()
        root = candidate.parent
        report = json.loads((candidate / "report.json").read_text())
        request = json.loads((candidate / "request.json").read_text())
        qualification = report["backend_qualification"]
        receipt_path = root / "qualification/comparison/qualification.json"
        receipt = json.loads(receipt_path.read_text())
        code_path = root / "Libs/population_dynamics" / fixtures.analysis._SOURCE_CODE_FILES["executor_sha256"]
        code_path.write_text("# Different engineering fixture backend implementation\n")
        code = dict(report["code"], executor_sha256=sha256_file(code_path))
        receipt.update(backend="cuda", numpy_version="other_fixture_numpy",
                       executor_sha256=code["executor_sha256"])
        self.fixture.write_json(receipt_path, receipt)
        delivery_path = root / "qualification/delivery.json"
        delivery = json.loads(delivery_path.read_text())
        delivery["files"] = [self.fixture.file_row(root / "qualification", "comparison/qualification.json")]
        self.fixture.write_json(delivery_path, delivery)
        qualification.update(qualification=receipt, receipt_sha256=sha256_file(receipt_path),
                             delivery_sha256=sha256_file(delivery_path))
        for item in (report, request):
            item.update(backend="cuda", code=code, backend_qualification=qualification)
        report["driver"]["numpy_version"] = receipt["numpy_version"]
        self.fixture.write_json(candidate / "request.json", request)
        self.fixture.write_json(candidate / "report.json", report)
        self.fixture.rebind(candidate)
        self.fixture.parameterize_archive(candidate)
        result = self.run_comparison(candidate, reference)
        self.assertEqual(result["status"], "COMPLETE_RECORDED_ARRAYS_IDENTICAL")
        self.assertNotEqual(result["source_bindings"]["candidate"]["producer_code"],
                            result["source_bindings"]["reference"]["producer_code"])
        self.assertEqual(result["source_bindings"]["candidate"]["backend"], "cuda")

    def staged(self, suffix):
        candidate, reference = self.pair(suffix)
        staged_candidate = self.fixture.root / ("staged_candidate" + suffix)
        staged_reference = self.fixture.root / ("staged_reference" + suffix)
        rows = []
        for source, staged in ((candidate, staged_candidate), (reference, staged_reference)):
            shutil.copytree(source.parent, staged)
            rows.append(dict(staged=str(staged), archive=str(source.parent),
                             case_id="pfc_94001_strong_cue",
                             delivery_sha256=sha256_file(source.parent / "delivery.json")))
        table = self.fixture.root / ("origins" + suffix + ".json")
        self.fixture.write_json(table, rows)
        return candidate, reference, staged_candidate, staged_reference, table

    def test_staged_inputs_require_matching_existing_durable_delivery(self):
        candidate, _, staged_candidate, staged_reference, table = self.staged("_valid")
        report = self.run_comparison(staged_candidate, staged_reference, "valid_origin", input_origins=table)
        self.assertEqual(report["source_bindings"]["candidate"]["archive"], str(candidate.parent.resolve()))
        self.assertEqual(report["source_bindings"]["candidate"]["origin_status"], "DURABLE_DELIVERY_VERIFIED")
        for mode in ("missing", "wrong"):
            with self.subTest(mode=mode):
                candidate, _, staged_candidate, staged_reference, table = self.staged("_" + mode)
                receipt = candidate.parent / "delivery.json"
                if mode == "missing":
                    receipt.unlink()
                else:
                    receipt.write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "DURABLE_DELIVERY"):
                    self.run_comparison(staged_candidate, staged_reference, mode, input_origins=table)


if __name__ == "__main__":
    unittest.main()

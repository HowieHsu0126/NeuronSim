"""Native clock coverage is read from the exact replay-verified source report."""
import json
import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from .pfc_stream_validation import native_clock_scope, _native_kernel_parameters, reuse_completed_phases
from . import pfc_native_drive, pfc_deterministic_replay
import numpy as np
from . import pfc_kernel_parameters as kernel_parameters
from .provenance import sha256_file


class NativeClockScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'capture/native/report.json'
        self.path.parent.mkdir(parents=True)

    def capture(self,clock,kernel=None,*,bound=True):
        config = dict(dt_ms=clock, **({} if kernel is None else kernel))
        native = dict(resolved_config=config)
        if kernel is not None and bound:
            native.update(resolved_kernel_parameters=kernel_parameters.resolve_kernel_parameters(config),
                          kernel_parameterization_sha256=sha256_file(Path(kernel_parameters.__file__)))
        self.path.write_text(json.dumps(native))
        return dict(status='PASS_NATIVE_NUMPY_DRIVE_REPLAY',consumed_files=[
            dict(path='capture/native/report.json',bytes=self.path.stat().st_size,sha256=sha256_file(self.path))])

    def test_both_native_policies_must_cover_the_same_clock(self):
        record=self.capture(.025)
        self.assertEqual(native_clock_scope(self.root,record,dict(dt_ms=.025)),[.025])
        with self.assertRaisesRegex(ValueError,'CLOCKS_DIFFER'):
            native_clock_scope(self.root,record,dict(dt_ms=.05))

    def test_unbound_or_changed_native_report_is_rejected(self):
        record=self.capture(.025)
        self.path.write_text(json.dumps(dict(resolved_config=dict(dt_ms=.05))))
        with self.assertRaisesRegex(ValueError,'BINDING_INVALID'):
            native_clock_scope(self.root,record,dict(dt_ms=.05))
        record=self.capture(.025);record['consumed_files']=[]
        with self.assertRaisesRegex(ValueError,'BINDING_INVALID'):
            native_clock_scope(self.root,record,dict(dt_ms=.025))

    def test_invalid_native_clock_is_not_qualified(self):
        for value in (True,0.,-.05):
            with self.subTest(value=value):
                record=self.capture(value)
                with self.assertRaisesRegex(ValueError,'CLOCKS_DIFFER'):
                    native_clock_scope(self.root,record,dict(dt_ms=value))



    def test_native_policies_must_match_resolved_kernel(self):
        kernel = dict(sigma_weight_profile=30., Jpos_excit2excit=1.7)
        result = self.capture(.05, kernel)
        self.assertEqual(native_clock_scope(self.root, result, dict(dt_ms=.05, **kernel)), [.05])
        with self.assertRaisesRegex(ValueError, 'KERNELS_DIFFER'):
            native_clock_scope(self.root, result, dict(dt_ms=.05))
        result = self.capture(.05)
        self.assertEqual(native_clock_scope(self.root, result,
            dict(dt_ms=.05, sigma_weight_profile=20., Jpos_excit2excit=1.6)), [.05])

    def test_nondefault_native_config_requires_parameterized_producer_binding(self):
        kernel = dict(sigma_weight_profile=30., Jpos_excit2excit=1.7)
        result = self.capture(.05, kernel, bound=False)
        with self.assertRaisesRegex(ValueError, 'NATIVE_KERNEL_BINDING_INVALID'):
            native_clock_scope(self.root, result, dict(dt_ms=.05, **kernel))
        with self.assertRaisesRegex(ValueError, 'NATIVE_KERNEL_BINDING_INVALID'):
            _native_kernel_parameters(dict(resolved_config=dict(dt_ms=.05, **kernel)))
        self.capture(.05, kernel)
        native = json.loads(self.path.read_text())
        native['kernel_parameterization_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'NATIVE_KERNEL_BINDING_INVALID'):
            _native_kernel_parameters(native)


class CompletedPhaseReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.validation = self.root / "validation"
        self.previous = self.root / "previous"
        self.pairing = self.root / "pairing.json"
        self.output = self.root / "output"
        self.case_ids = ["case_" + str(i) for i in range(8)]
        config = dict(dt_ms=0.0125, cases=[dict(case_id=name) for name in self.case_ids])
        native = self.write(self.validation / "capture/native/report.json", dict(resolved_config=config))
        entry = dict(path="capture/native/report.json", bytes=native.stat().st_size, sha256=sha256_file(native))
        self.write(self.validation / "delivery.json", dict(status="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY", files=[entry]))
        dsha = sha256_file(self.validation / "delivery.json")
        self.write(self.pairing, dict(status="PASS_NATIVE_INPUT_PAIRING", delivery_sha256=dsha))
        psha = sha256_file(self.pairing)
        inp, rec = self.previous / "input_replay", self.previous / "recorded_regression"
        common = dict(delivery_sha256=dsha, pairing_sha256=psha, consumed_files=[entry])
        inputs = dict(common, status="PASS_NATIVE_NUMPY_DRIVE_REPLAY", completed_comparisons=24,
                      passed_comparisons=24, numpy_version=np.__version__, source_entry_sha256=pfc_native_drive.SOURCE_ENTRY_SHA256,
                      implementation_sha256=sha256_file(Path(pfc_native_drive.__file__)), cases=[])
        recorded = dict(common, status="PASS_PFC_DETERMINISTIC_REPLAY", backend="numpy", completed_cases=8,
                        passed_cases=8, not_completed_cases=0, resolved_config=config,
                        implementation_sha256=sha256_file(Path(pfc_deterministic_replay.__file__)), cases=[])
        input_table, recorded_table = [], []
        for name in self.case_ids:
            for block in (1, 37, 256):
                stem = name + "_" + str(block)
                rng = inp / (stem + ".rng")
                rng.parent.mkdir(parents=True, exist_ok=True)
                rng.write_bytes(b"fixture rng")
                case = dict(case_id=name, block_rows=block, passed=True,
                            generated_rng_file=rng.name, generated_rng_sha256=sha256_file(rng))
                receipt = self.write(inp / (stem + ".json"), case)
                inputs["cases"].append(dict(case, receipt_file=receipt.name, receipt_sha256=sha256_file(receipt)))
                input_table += [dict(case_id=name, block_rows=block, population=pop, passed=True) for pop in ("E", "I")]
            folder = rec / name
            folder.mkdir(parents=True)
            raw, terminal = folder / "raw.npz", folder / "terminal_state.npz"
            raw.write_bytes(b"fixture raw")
            terminal.write_bytes(b"fixture terminal")
            case = dict(case_id=name, status="PASS_NUMERICAL_SOURCE_COMPARISON",
                        implementation_sha256=recorded["implementation_sha256"], pairing_sha256=psha, delivery_sha256=dsha,
                        raw_file=raw.name, raw_sha256=sha256_file(raw), terminal_state_file=terminal.name,
                        terminal_state_sha256=sha256_file(terminal), comparisons=[dict(passed=True)])
            receipt = self.write(folder / "comparison.json", case)
            recorded["cases"].append(dict(case_id=name, status=case["status"], report_file=name + "/comparison.json",
                                           report_sha256=sha256_file(receipt)))
            recorded_table.append(dict(case_id=name, passed=True))
        self.table(inp / "drive_comparisons.csv", input_table)
        self.table(rec / "array_comparisons.csv", recorded_table)
        inputs["comparison_table_sha256"] = sha256_file(inp / "drive_comparisons.csv")
        recorded["array_comparisons_sha256"] = sha256_file(rec / "array_comparisons.csv")
        self.write(inp / "comparison.json", inputs)
        self.write(rec / "comparison.json", recorded)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return path

    @staticmethod
    def table(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def reuse(self):
        return reuse_completed_phases(self.validation, self.pairing, self.previous, self.output, "numpy")

    def test_complete_phases_copy_all_results_without_execution(self):
        inputs, recorded, reuse = self.reuse()
        self.assertEqual(inputs["passed_comparisons"], 24)
        self.assertEqual(recorded["passed_cases"], 8)
        self.assertEqual(reuse["previous_phase_simulations_repeated"], 0)
        self.assertTrue((self.output / "recorded_regression/case_7/raw.npz").exists())
        for row in reuse["reused_outputs"]:
            self.assertEqual(sha256_file(self.output / row["path"]), row["sha256"])

    def test_duplicate_case_cannot_replace_missing_case(self):
        path = self.previous / "input_replay/comparison.json"
        value = json.loads(path.read_text())
        value["cases"][-1] = deepcopy(value["cases"][0])
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "DENOMINATOR_CHANGED"):
            self.reuse()

    def test_same_size_changed_source_rejects_reuse(self):
        path = self.validation / "capture/native/report.json"
        data = path.read_bytes()
        path.write_bytes(data.replace(b"case_0", b"case_9", 1))
        with self.assertRaisesRegex(ValueError, "SOURCE_LINEAGE_CHANGED"):
            self.reuse()

    def test_failed_table_row_cannot_be_hidden_by_success_summary(self):
        table = self.previous / "input_replay/drive_comparisons.csv"
        table.write_text(table.read_text().replace("True", "False", 1))
        path = self.previous / "input_replay/comparison.json"
        value = json.loads(path.read_text())
        value["comparison_table_sha256"] = sha256_file(table)
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "INPUT_TABLE_CHANGED"):
            self.reuse()

    def test_changed_result_and_producer_reject_reuse(self):
        path = self.previous / "recorded_regression/case_0/raw.npz"
        path.write_bytes(b"changed raw")
        with self.assertRaisesRegex(ValueError, "OUTPUT_CHANGED"):
            self.reuse()
        path = self.previous / "recorded_regression/comparison.json"
        value = json.loads(path.read_text())
        value["implementation_sha256"] = "0" * 64
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "STATUS_OR_PRODUCER_CHANGED"):
            self.reuse()


if __name__=='__main__':
    unittest.main()

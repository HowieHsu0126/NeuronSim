"""Native-reference adapter tests using the existing complete engineered archive."""
import json
from pathlib import Path
import unittest

from . import pfc_native_fluctuation_analysis as analysis
from . import pfc_temporal_readouts as temporal
from . import test_pfc_temporal_readouts as fixtures
from .provenance import sha256_file


class NativeFluctuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeReferenceTemporalTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def prepare(self, terminal=False):
        self.fixture.make_archive(terminal=terminal)
        self.temporal = Path(self.fixture.temporary.name)/"temporal"
        temporal.analyze_native_reference(self.fixture.root, self.temporal/"analysis")
        path = self.temporal/"analysis/analysis.json"
        (self.temporal/"delivery.json").write_text(json.dumps(dict(
            status="ARCHIVED_PFC_SOURCE_READOUTS", exit_code=0, readout_mode="reference-temporal",
            files=[dict(path="analysis/analysis.json", bytes=path.stat().st_size, sha256=sha256_file(path))])))
        # The original scratch location is deliberately unusable.
        report = json.loads(path.read_text());report["source_reference_archive"]="/deleted/scratch/native_reference"
        path.write_text(json.dumps(report))
        receipt = json.loads((self.temporal/"delivery.json").read_text())
        receipt["files"][0].update(bytes=path.stat().st_size, sha256=sha256_file(path))
        (self.temporal/"delivery.json").write_text(json.dumps(receipt))

    def test_cli_uses_bound_complete_native_root_and_retains_all_units(self):
        self.prepare()
        output = Path(self.fixture.temporary.name)/"fluctuations"
        self.assertEqual(analysis.main(["--native-reference-archive", str(self.fixture.root),
            "--reference-readouts", str(self.temporal), "--output", str(output)]), 0)
        result = json.loads((output/"report.json").read_text())
        self.assertEqual(result["status"], "COMPLETE_SOURCE_FLUCTUATION_CALIBRATION")
        self.assertEqual(result["native_report_path"], "native_recovery/report.json")
        self.assertEqual(result["completed_source_cases"], 9)
        self.assertEqual(result["coupled_core_units"], 2560)
        self.assertEqual(result["unit_bin_count_values_consumed"], 9*1000*2560)
        self.assertEqual(result["field_unit_counts"], [128]*16+[512])
        self.assertEqual(len(result["population_summaries"]), 45)
        self.assertEqual(result["terminal_tick_events_excluded"], 0)
        self.assertFalse(result["scientific_fidelity_qualified"])
        self.assertFalse(result["finite_size_scaling_qualified"])

    def test_terminal_event_remains_explicitly_outside_half_open_counts(self):
        self.prepare(terminal=True)
        output = Path(self.fixture.temporary.name)/"terminal"
        result = analysis.run(self.fixture.root, self.temporal, output)
        self.assertEqual(result["terminal_tick_events_excluded"], 1)
        self.assertEqual(sum(row["terminal_tick_events_excluded"] for row in result["source_bindings"]), 1)

    def test_changed_bound_native_raw_data_is_rejected(self):
        self.prepare()
        prior = json.loads((self.temporal/"analysis/analysis.json").read_text())
        raw = self.fixture.root/prior["completed_cases"][0]["source_binding"]["raw_path"]
        with raw.open("ab") as stream:
            stream.write(b"changed")
        output = Path(self.fixture.temporary.name)/"corrupt"
        with self.assertRaisesRegex(ValueError, "size or hash mismatch"):
            analysis.run(self.fixture.root, self.temporal, output)
        self.assertEqual(json.loads((output/"report.json").read_text())["status"], "FAILED")

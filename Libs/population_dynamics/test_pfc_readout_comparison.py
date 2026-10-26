"""Disposable schema fixtures; these archives are not scientific evidence."""
import csv
import hashlib
import subprocess
import sys
import json
import tempfile
import unittest
from pathlib import Path

from . import pfc_readout_comparison as pfc
from .provenance import sha256_file


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal(root):
    area = root / "analysis"
    analysis = json.loads((area / "analysis.json").read_text())
    artifacts = {"responses.csv": sha256_file(area / "responses.csv"), "unread_array.npz": "e" * 64}
    _json(area / "readout_replay.json", {"status": "PASS", "complete_episodes": 9,
                                       "units": analysis["units"], "artifacts": artifacts})
    analysis.update(artifacts=artifacts, readout_replay_sha256=sha256_file(area / "readout_replay.json"))
    _json(area / "analysis.json", analysis)
    _json(root / "delivery.json", {"files": [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in pfc.CONSUMED]})


def _fixture(root, expanded=False):
    area = root / "analysis"
    area.mkdir(parents=True)
    seeds = [94001, 94002, 94003] if expanded else [93001, 93002, 93003]
    ne, ni, factor = (80000, 20000, 0.0256) if expanded else (2048, 512, 1.0)
    cases = [{"case_id": f"{seed}_{condition}", "source_rng_seed": seed,
              "stimulus_strength_nA": strength, "distractor_strength_nA": 0}
             for seed in seeds for strength, condition in pfc.CONDITIONS.items()]
    config = dict(pfc.PHYSICAL, excitatory_neurons=ne, inhibitory_neurons=ni,
                  weight_scaling_factor=factor, cases=cases)
    analysis = {"status": "DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC" if expanded else
                "DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC", "units": ne + ni,
                "complete_source_episodes": 9, "source_revision": pfc.SOURCE_REVISION,
                "source_entry_sha256": pfc.SOURCE_ENTRY_SHA256, "source_rng_seeds": seeds,
                "phases_ms": pfc.PHASES, "configuration": config,
                "common_execution_binding": {"backend": "cuda" if expanded else "brian2"}}
    rows = []
    for case in cases:
        value = case["source_rng_seed"] - seeds[0] + 1
        for phase, (start, stop) in pfc.PHASES.items():
            row = dict.fromkeys(pfc.COLUMNS, "0")
            row.update(case_id=case["case_id"], source_rng_seed=case["source_rng_seed"],
                       condition=pfc.CONDITIONS[case["stimulus_strength_nA"]], phase=phase,
                       start_ms=start, stop_ms=stop, units=ne + ni,
                       excitatory_neurons=ne, inhibitory_neurons=ni,
                       stimulus_strength_nA=case["stimulus_strength_nA"],
                       excitatory_rate_hz=value, inhibitory_rate_hz=value,
                       cue_selectivity_hz=value, resultant_length=value / 10, direction_defined="True")
            rows.append(row)
    pfc._write_csv(area / "responses.csv", pfc.COLUMNS, rows)
    _json(area / "analysis.json", analysis)
    _seal(root)
    return root


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.reference = _fixture(self.base / "reference")
        self.expanded = _fixture(self.base / "expanded", True)
        self.output = self.base / "comparison"

    def test_complete_unpaired_comparison_with_reordered_columns(self):
        path = self.expanded / "analysis/responses.csv"
        rows = list(csv.DictReader(path.read_text().splitlines()))
        pfc._write_csv(path, list(reversed(pfc.COLUMNS)), rows)
        _seal(self.expanded)
        report = pfc.compare(self.reference, self.expanded, self.output)
        rows = list(csv.DictReader((self.output / "combined.csv").read_text().splitlines()))
        summary = list(csv.DictReader((self.output / "descriptive.csv").read_text().splitlines()))
        self.assertEqual(len(rows), 90)
        self.assertEqual({row["cohort"] for row in rows}, {"reference", "expanded"})
        self.assertEqual(len(summary), 120)
        self.assertEqual({row["n"] for row in summary}, {"3"})
        sample = next(row for row in summary if row["metric"] == "excitatory_rate_hz")
        self.assertEqual([float(sample[key]) for key in ("mean", "min", "max")], [2, 1, 3])
        self.assertFalse(report["paired_across_sizes"])
        self.assertFalse(report["source_scaling_qualified"])
        self.assertEqual(report["source_episodes_reused"], 18)

    def test_missing_or_duplicate_phase_rejected_after_resealing(self):
        path = self.reference / "analysis/responses.csv"
        original = list(csv.DictReader(path.read_text().splitlines()))
        for rows in (original[:-1], original[:-1] + [original[0]]):
            pfc._write_csv(path, pfc.COLUMNS, rows)
            _seal(self.reference)
            with self.assertRaises(ValueError):
                pfc.compare(self.reference, self.expanded, self.output)

    def test_each_changed_consumed_file_is_rejected(self):
        for name in pfc.CONSUMED:
            path = self.reference / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                pfc.compare(self.reference, self.expanded, self.output)
            path.write_bytes(original)

    def test_timing_configuration_and_normalization_rejected(self):
        for kind in ("timing", "configuration", "normalization"):
            reference = _fixture(self.base / kind)
            if kind == "timing":
                path = reference / "analysis/responses.csv"
                rows = list(csv.DictReader(path.read_text().splitlines()))
                rows[0]["stop_ms"] = "99"
                pfc._write_csv(path, pfc.COLUMNS, rows)
            else:
                path = reference / "analysis/analysis.json"
                analysis = json.loads(path.read_text())
                key = "stimulus_start_ms" if kind == "configuration" else "weight_scaling_factor"
                analysis["configuration"][key] += 1
                _json(path, analysis)
            _seal(reference)
            with self.assertRaises(ValueError):
                pfc.compare(reference, self.expanded, self.output)



def _clock_seal(root):
    area = root / "analysis"
    analysis = json.loads((area / "analysis.json").read_text())
    rows = list(csv.DictReader((area / "responses.csv").read_text().splitlines()))
    ids = {row["case_id"] for row in rows}
    complete = len(ids)
    analysis.update(complete_source_episodes=complete, registered_source_cases=9,
        missing_source_cases=[case["case_id"] for case in analysis["configuration"]["cases"] if case["case_id"] not in ids],
        source_bindings=[], phase_unit_count_values_verified=complete * analysis["units"] * len(pfc.PHASES),
        config_sha256=hashlib.sha256(json.dumps(analysis["configuration"], sort_keys=True).encode()).hexdigest())
    artifacts = {"responses.csv": sha256_file(area / "responses.csv")}
    replay = dict(status="PASS", complete_episodes=complete, registered_source_cases=9, units=analysis["units"],
        source_bindings=[], config_sha256=analysis["config_sha256"], artifacts=artifacts,
        phase_unit_count_values_checked=analysis["phase_unit_count_values_verified"])
    _json(area / "readout_replay.json", replay)
    analysis.update(artifacts=artifacts, readout_replay_sha256=sha256_file(area / "readout_replay.json"))
    _json(area / "analysis.json", analysis)
    _json(root / "delivery.json", {"files": [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in pfc.CONSUMED]})


def _clock_fixture(root, dt, *, condition="strong_cue", rate=1, angle=0):
    _fixture(root, expanded=True)
    area = root / "analysis"
    analysis = json.loads((area / "analysis.json").read_text())
    analysis["configuration"]["dt_ms"] = dt
    analysis.update(status="PARTIAL_SOURCE_EXECUTION_DIAGNOSTIC", failed_source_attempts=[], unresolved_input_paths=[])
    driver = dict(binomial_n=1000, external_rate_Hz=1.4, dt_seconds=dt / 1000,
        binomial_probability=1.4 * (dt / 1000), total_ticks=round(5000 / dt), ticks_generated=round(5000 / dt),
        n_e=80000, n_i=20000, initial_draw_order="I_then_E", initial_voltage_archive_order="E_then_I",
        packed_draw_order="tick_major_I_then_E", random_stream_policy="source_global", returned_dtype="uint16",
        rng="numpy.random.RandomState(MT19937)", numpy_version="fixture_numpy", source_entry_sha256=pfc.SOURCE_ENTRY_SHA256)
    analysis["common_execution_binding"] = dict(backend="numpy", driver=driver,
        code=dict(executor_sha256="a" * 64, drive_sha256="b" * 64, runner_sha256=str(dt)))
    rows = [row for row in csv.DictReader((area / "responses.csv").read_text().splitlines())
            if row["case_id"] == f"94001_{condition}"]
    for row in rows:
        row.update(excitatory_rate_hz=rate, circular_cue_error_deg=angle)
    pfc._write_csv(area / "responses.csv", pfc.COLUMNS, rows)
    _json(area / "analysis.json", analysis)
    _clock_seal(root)
    return root


class ClockComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.archives = [_clock_fixture(self.base / str(dt), dt, rate=3-i, angle=angle)
                         for i, (dt, angle) in enumerate(zip(pfc.CLOCKS_MS, (179, -179, 178)))]
        self.output = self.base / "comparison"

    def mutate_analysis(self, index, change):
        path = self.archives[index] / "analysis/analysis.json"
        value = json.loads(path.read_text()); change(value); _json(path, value)
        _clock_seal(self.archives[index])

    def test_clock_order_differences_and_partial_denominator(self):
        report = pfc.compare_clocks(list(reversed(self.archives)), self.output)
        self.assertEqual(report["dt_ms"], list(pfc.CLOCKS_MS))
        self.assertEqual(report["completed_cases_per_clock"], 1)
        self.assertEqual(report["registered_cases_per_clock"], 9)
        self.assertEqual(report["source_episodes_reused"], 3)
        self.assertEqual(report["source_rng_blocks_compared"], [94001])
        self.assertFalse(report["full_registered_case_coverage"])
        self.assertFalse(report["matched_physical_input_paths"])
        self.assertFalse(report["distributional_convergence_qualified"])
        self.assertIsNone(report["p_values"])
        self.assertEqual(len(report["inputs"][0]["missing_case_ids"]), 8)
        differences = list(csv.DictReader((self.output / "adjacent_clock_differences.csv").read_text().splitlines()))
        self.assertEqual(len(differences), 60)
        self.assertTrue(all(float(row["coarse_minus_fine"]) == 1 for row in differences
                            if row["metric"] == "excitatory_rate_hz"))
        angles = [float(row["coarse_minus_fine"]) for row in differences
                  if row["metric"] == "circular_cue_error_deg"]
        self.assertEqual(angles, [-2] * 5 + [3] * 5)

    def test_duplicate_clock_and_changed_physical_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate registered clock"):
            pfc.compare_clocks([self.archives[0], self.archives[0], self.archives[2]], self.output)
        self.mutate_analysis(1, lambda value: value["configuration"].update(stimulus_width_deg=31))
        with self.assertRaisesRegex(ValueError, "physical protocol"):
            pfc.compare_clocks(self.archives, self.output)

    def test_mismatched_case_sets_cannot_be_reduced_to_intersection(self):
        other = _clock_fixture(self.base / "other_condition", .025, condition="cue")
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            pfc.compare_clocks([self.archives[0], other, self.archives[2]], self.output)

    def test_native_input_law_and_dynamics_identity_rejected_after_resealing(self):
        self.mutate_analysis(1, lambda value: value["common_execution_binding"]["driver"].update(binomial_probability=.1))
        with self.assertRaisesRegex(ValueError, "input law"):
            pfc.compare_clocks(self.archives, self.output)
        self.mutate_analysis(1, lambda value: value["common_execution_binding"]["driver"].update(binomial_probability=1.4 * (.025 / 1000)))
        self.mutate_analysis(1, lambda value: value["common_execution_binding"]["code"].update(executor_sha256="c" * 64))
        with self.assertRaisesRegex(ValueError, "dynamics implementation"):
            pfc.compare_clocks(self.archives, self.output)

    def test_incomplete_phase_rows_rejected_after_resealing(self):
        path = self.archives[1] / "analysis/responses.csv"
        rows = list(csv.DictReader(path.read_text().splitlines()))
        pfc._write_csv(path, pfc.COLUMNS, rows[:-1]); _clock_seal(self.archives[1])
        with self.assertRaisesRegex(ValueError, "all phases"):
            pfc.compare_clocks(self.archives, self.output)

    def test_undefined_direction_stays_missing(self):
        path = self.archives[1] / "analysis/responses.csv"
        rows = list(csv.DictReader(path.read_text().splitlines()))
        rows[0].update(direction_defined="False", decoded_angle_deg="", circular_cue_error_deg="")
        pfc._write_csv(path, pfc.COLUMNS, rows); _clock_seal(self.archives[1])
        pfc.compare_clocks(self.archives, self.output)
        differences = list(csv.DictReader((self.output / "adjacent_clock_differences.csv").read_text().splitlines()))
        missing = [row for row in differences if row["comparison_status"] == "UNDEFINED_DIRECTION"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(row["coarse_minus_fine"] == "" for row in missing))

    def test_changed_consumed_readout_rejected(self):
        path = self.archives[0] / "analysis/responses.csv"
        path.write_bytes(path.read_bytes() + b"\\n")
        with self.assertRaisesRegex(ValueError, "Clock readout changed"):
            pfc.compare_clocks(self.archives, self.output)

    def test_real_cli_accepts_clock_mode_and_rejects_mixed_modes(self):
        command = [sys.executable, "-B", "-m", pfc.__name__, "--output", str(self.output)]
        for root in self.archives:
            command += ["--clock-readout", str(root)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(command + ["--reference", str(self.archives[0])], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose size comparison or clock comparison", result.stderr)

if __name__ == "__main__":
    unittest.main()

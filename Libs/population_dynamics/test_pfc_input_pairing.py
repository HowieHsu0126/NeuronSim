"""Synthetic contract tests for source-input pairing; no native simulation.

Tiny arrays have hand-counted differences. Fixture status labels exercise the
archived format and are not source evidence. Run tests on a compute node.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .pfc_input_pairing import analyze


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _PairingFixture:
    """A full factorial fixture with deliberately distinct, known realizations."""

    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.paths = set()
        self.case_ids, self.receipts = {}, {}
        self.source_hash = "1" * 64
        self.config = dict(scope="source_reproduction", random_stream_policy="independent_initial_external",
                           expected_source_revision="synthetic_test_fixture", stimulus_strength_nA=0.06,
                           excitatory_neurons=3, inhibitory_neurons=2, dt_ms=0.05, duration_ms=0.2, cases=[])
        self.inputs = dict(status="NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE", mode="capture",
                           draw_capture_performed=True, source_entry_sha256=self.source_hash,
                           brian2_version="2.5.4", numpy_version=np.__version__, cases=[])
        self.native = dict(status="NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS",
                           random_stream_policy="independent_initial_external",
                           source_entry_sha256=self.source_hash, source_revision="synthetic_test_fixture",
                           brian2_version="2.5.4", numpy_version=np.__version__, resolved_config=self.config, cases=[])
        self.execution = dict(status="RECORDED_NATIVE_PFC_INPUT_OBSERVATION", mode="capture", cases=8)
        self.comparison = dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",
                               captured_draw_values_recounted=160, initial_voltage_values_compared=40, cases=[])
        for initial in (95001, 95002):
            for external in (96001, 96002):
                for control in (0.0, 0.06):
                    label = "omission" if control == 0 else "cue"
                    case_id = f"initial_{initial}_external_{external}_{label}"
                    self.case_ids[(initial, external, control)] = case_id
                    row = dict(case_id=case_id, initial_rng_seed=initial, external_noise_seed=external,
                               stimulus_strength_nA=control, distractor_strength_nA=0.0)
                    self.config["cases"].append(row)
                    resolved = dict(row, source_rng_seed=external,
                                    random_stream_policy="independent_initial_external",
                                    block_id=f"pfc_initial_{initial}_external_{external}")
                    # Seed 96001 totals E=13, I=6. Seed 96002 totals E=16, I=5.
                    # Exactly two entries differ in each population, not all cells.
                    e = np.array([[0, 1, 2], [3, 0, 1], [0, 2, 0], [1, 0, 3]], dtype=np.uint16)
                    i = np.array([[0, 1], [2, 0], [1, 0], [0, 2]], dtype=np.uint16)
                    if external == 96002:
                        e[0, 0], e[2, 1] = 4, 1
                        i[1, 0], i[3, 1] = 0, 3
                    initial_values = np.array([-59.5, -58.5, -57.5, -56.5, -55.5], dtype=np.float64)
                    if initial == 95002:
                        initial_values[2], initial_values[4] = -57.25, -54.75
                    initial_file = case_id + "_initial_voltage.npy"
                    self.save_array("capture/native/" + initial_file, initial_values)
                    self.native["cases"].append(dict(row, source_rng_seed=external,
                        source_rng_seed_role="external_noise", core_units=5, initial_voltage_file=initial_file))
                    providers = []
                    for population, values in (("I", i), ("E", e)):
                        name = case_id + "/" + population + "_external_counts.npy"
                        self.save_array("capture/inputs/" + name, values)
                        providers.append(dict(population=population, counts_file=name,
                            counts_shape=[4, len(values[0])], units=len(values[0]), expected_sample_rows=4,
                            observed_sampler_calls=4, dt_seconds=0.05 / 1000, terminal_clock_seconds=0.2 / 1000,
                            target_variable="s_AMPA", N=1000, rate_Hz=1.4, probability=1.4 * (0.05 / 1000),
                            weight_dimensionless=1.0, when="synapses", order=0,
                            source_unit_order="ascending_zero_based_population_index"))
                    self.receipts[case_id] = dict(status="NATIVE_INPUT_OBSERVATION_CASE_COMPLETE",
                        mode="capture", case_id=case_id, case=resolved, source_entry_sha256=self.source_hash,
                        source_population_archive_order=["E", "I"], source_provider_construction_order=["I", "E"],
                        providers=providers)
                    self.inputs["cases"].append(dict(case_id=case_id, mode="capture",
                                                     receipt_file=case_id + "/observation.json"))
                    self.comparison["cases"].append(dict(case_id=case_id, terminal_states_equal=True,
                                                         numpy_rng_equal=True, python_rng_equal=True))
        self.publish()

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        self.paths.add(name)

    def save_array(self, name, values):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        self.paths.add(name)

    def count_path(self, initial, external, control, population):
        case_id = self.case_ids[(initial, external, control)]
        return self.root / "capture/inputs" / case_id / (population + "_external_counts.npy")

    def initial_path(self, initial, external, control):
        case_id = self.case_ids[(initial, external, control)]
        return self.root / "capture/native" / (case_id + "_initial_voltage.npy")

    def replace(self, path, values):
        self.save_array(path.relative_to(self.root).as_posix(), values)

    def publish(self):
        """Rebind the fixture completely; numerical violations stay detectable."""
        for observed, native, prior in zip(self.inputs["cases"], self.native["cases"], self.comparison["cases"]):
            receipt = self.receipts[observed["case_id"]]
            for provider in receipt["providers"]:
                path = self.root / "capture/inputs" / provider["counts_file"]
                provider["counts_sha256"] = _digest(path)
                provider["captured_external_events"] = int(np.load(path, allow_pickle=False).sum())
            prior["external_events"] = sum(row["captured_external_events"] for row in receipt["providers"])
            native["initial_voltage_sha256"] = _digest(self.root / "capture/native" / native["initial_voltage_file"])
            receipt_name = "capture/inputs/" + observed["receipt_file"]
            self.write_json(receipt_name, receipt)
            observed["receipt_sha256"] = _digest(self.root / receipt_name)
        self.write_json("Libs/configs/fixture.yaml", self.config)  # JSON is valid YAML.
        self.write_json("capture/inputs/report.json", self.inputs)
        self.write_json("capture/native/report.json", self.native)
        self.execution.update(config_sha256=_digest(self.root / "Libs/configs/fixture.yaml"),
                              native_report_sha256=_digest(self.root / "capture/native/report.json"),
                              observation_report_sha256=_digest(self.root / "capture/inputs/report.json"))
        self.write_json("capture/execution.json", self.execution)
        self.comparison["capture_execution_sha256"] = _digest(self.root / "capture/execution.json")
        self.write_json("comparison/comparison.json", self.comparison)
        files = [dict(path=name, sha256=_digest(self.root / name), bytes=(self.root / name).stat().st_size)
                 for name in sorted(self.paths) if name != "delivery.json"]
        self.write_json("delivery.json", dict(status="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY",
                                               files=files, comparison=self.comparison))


class InputPairingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = _PairingFixture(self.root / "validation")

    def run_analysis(self):
        return analyze(self.fixture.root, self.root / "analysis")

    def assert_complete_comparisons(self, report):
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual(len(report["comparisons"]), 30)
        self.assertEqual(report["denominator"]["completed_comparisons"],
                         {"within_external_seed": 12, "within_initial_seed": 6,
                          "between_external_seeds": 8, "between_initial_seeds": 4})
        self.assertTrue(report["input_identity_mapping_complete"])

    def alter_fixed_seed_draw(self):
        path = self.fixture.count_path(95002, 96001, 0.06, "E")
        values = np.load(path, allow_pickle=False)
        values[1, 1] = 1  # Zero in both unmodified drive realizations.
        self.fixture.replace(path, values)

    def collapse_external_realizations(self):
        for initial in (95001, 95002):
            for control in (0.0, 0.06):
                for population in ("E", "I"):
                    source = self.fixture.count_path(initial, 96001, control, population)
                    target = self.fixture.count_path(initial, 96002, control, population)
                    self.fixture.replace(target, np.load(source, allow_pickle=False))
        self.fixture.publish()

    def test_complete_factorial_has_hand_calculated_pair_counts_and_differences(self):
        report = self.run_analysis()
        self.assertEqual(report["status"], "PASS_NATIVE_INPUT_PAIRING")
        self.assert_complete_comparisons(report)
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["denominator"]["validated_count_arrays"], 16)
        self.assertEqual(report["denominator"]["validated_count_values"], 160)
        self.assertEqual(report["denominator"]["validated_initial_voltage_values"], 40)
        for row in report["comparisons"]:
            between = row["kind"].startswith("between_")
            self.assertEqual(row["mismatched_values"], 2 if between else 0)
            self.assertEqual(row["array_equal"], not between)
            expected_values = {"E": 12, "I": 8, "E_and_I_initial_voltage": 5}[row["population"]]
            self.assertEqual(row["values_compared"], expected_values)
            if row["kind"] == "between_external_seeds":
                self.assertEqual((row["left_external_events"], row["right_external_events"]),
                                 (13, 16) if row["population"] == "E" else (6, 5))
        self.assertEqual(sum(row["values_compared"] for row in report["comparisons"]), 250)
        with (self.root / "analysis/pair_comparisons.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(int(row["mismatched_values"]) for row in rows), 24)
        self.assertFalse(report["statistical_independence_established"])
        self.assertFalse(report["biological_independence_established"])
        self.assertEqual(report["production_episodes"], 0)

    def test_changed_consumed_count_hash_fails_with_full_planned_case_denominator(self):
        self.alter_fixed_seed_draw()  # Deliberately do not rebind the archive.
        report = self.run_analysis()
        self.assertEqual(report["status"], "FAILED_NATIVE_INPUT_PAIRING_INTEGRITY")
        self.assertIn("PFC_PAIRING_ARTIFACT_CHANGED", report["failure"]["message"])
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual(report["denominator"]["expected_cases"], 8)
        self.assertEqual(report["denominator"]["expected_count_arrays"], 16)
        self.assertLess(report["denominator"]["validated_cases"], 8)
        self.assertFalse(report["input_identity_mapping_complete"])
        saved = json.loads((self.root / "analysis/pairing.json").read_text())
        self.assertEqual(saved["failure"], report["failure"])

    def test_rebound_fixed_seed_draw_mismatch_keeps_every_comparison(self):
        self.alter_fixed_seed_draw()
        self.fixture.publish()
        report = self.run_analysis()
        self.assertEqual(report["status"], "FAIL_NATIVE_INPUT_PAIRING")
        self.assert_complete_comparisons(report)
        self.assertFalse(report["gates"]["fixed_external_seed_inputs_equal"])
        self.assertTrue(report["gates"]["fixed_initial_seed_voltages_equal"])
        failures = [row for row in report["comparisons"]
                    if row["kind"] == "within_external_seed" and not row["array_equal"]]
        self.assertEqual(len(failures), 1)
        self.assertEqual((failures[0]["population"], failures[0]["mismatched_values"]), ("E", 1))
        self.assertEqual(failures[0]["right_case"], self.fixture.case_ids[(95002, 96001, 0.06)])

    def test_collapsed_distinct_noise_seeds_fail_even_with_consistent_manifests(self):
        self.collapse_external_realizations()
        report = self.run_analysis()
        self.assertEqual(report["status"], "FAIL_NATIVE_INPUT_PAIRING")
        self.assert_complete_comparisons(report)
        self.assertTrue(report["gates"]["fixed_external_seed_inputs_equal"])
        self.assertFalse(report["gates"]["distinct_external_seed_joint_realizations"])
        rows = [row for row in report["comparisons"] if row["kind"] == "between_external_seeds"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["mismatched_values"] == 0 for row in rows))

    def test_multiple_draw_and_initial_pairing_failures_are_all_retained(self):
        self.alter_fixed_seed_draw()
        path = self.fixture.initial_path(95002, 96002, 0.0)
        initial = np.load(path, allow_pickle=False)
        initial[0] = -59.25
        self.fixture.replace(path, initial)
        self.fixture.publish()
        report = self.run_analysis()
        self.assertEqual(report["status"], "FAIL_NATIVE_INPUT_PAIRING")
        self.assert_complete_comparisons(report)
        self.assertFalse(report["gates"]["fixed_external_seed_inputs_equal"])
        self.assertFalse(report["gates"]["fixed_initial_seed_voltages_equal"])
        self.assertTrue(report["gates"]["distinct_external_seed_joint_realizations"])
        self.assertTrue(report["gates"]["distinct_initial_seed_realizations"])
        failing_kinds = {row["kind"] for row in report["comparisons"]
                         if row["kind"].startswith("within_") and not row["array_equal"]}
        self.assertEqual(failing_kinds, {"within_external_seed", "within_initial_seed"})
        saved = json.loads((self.root / "analysis/pairing.json").read_text())
        self.assertEqual(saved["comparisons"], report["comparisons"])

    def test_cli_returns_nonzero_and_archives_complete_negative_result(self):
        self.collapse_external_realizations()
        environment = dict(os.environ)
        libs = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = libs + (os.pathsep + environment["PYTHONPATH"]
                                           if environment.get("PYTHONPATH") else "")
        output = self.root / "cli_analysis"
        result = subprocess.run([sys.executable, "-m", "population_dynamics.pfc_input_pairing",
                                 "--validation", str(self.fixture.root), "--output", str(output)],
                                cwd=self.root, env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["status"], "FAIL_NATIVE_INPUT_PAIRING")
        report = json.loads((output / "pairing.json").read_text())
        self.assert_complete_comparisons(report)
        self.assertFalse(report["gates"]["distinct_external_seed_joint_realizations"])


if __name__ == "__main__":
    unittest.main()

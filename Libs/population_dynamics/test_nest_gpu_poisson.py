"""CPU-only contract tests for the native Poisson-driver diagnostic."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from Libs.population_dynamics import nest_gpu_poisson as subject


def base_config():
    return {
        "dt_ms": 0.1, "tau_syn_ms": 0.5, "delay_ms": 0.2,
        "rates_hz": [128.0], "targets_per_rate": 1,
        "duration_ms": 2000.0, "seed": 74011,
    }


def fp32_currents(counts, decay, unit_additions):
    counts = np.asarray(counts, dtype=np.int64)
    decay = np.asarray(decay, dtype=np.float32)
    current = np.empty((len(counts) + 1, counts.shape[1]), dtype=np.float32)
    current[0] = np.asarray([0.25, 1.5, 0.0], dtype=np.float32)
    for row, additions in enumerate(counts):
        accumulated = current[row].copy()
        if unit_additions:
            for event in range(int(additions.max())):
                mask = additions > event
                accumulated[mask] = np.float32(accumulated[mask] + np.float32(1.0))
        else:
            accumulated = np.float32(accumulated + additions.astype(np.float32))
        current[row + 1] = np.float32(accumulated * decay)
    return current.astype(np.float64)


class DecoderTests(unittest.TestCase):
    def test_recovers_aggregated_and_unit_additions_in_fp32(self):
        counts = np.tile(np.asarray([
            [0, 1, 2], [1, 0, 0], [7, 2, 1], [0, 0, 0],
            [2, 17, 3], [0, 1, 0], [1, 0, 4], [0, 0, 0],
        ], dtype=np.int64), (4, 1))
        decay = np.exp(np.asarray([-0.2, -0.1, -0.4], dtype=np.float32))
        for unit_additions in (False, True):
            with self.subTest(unit_additions=unit_additions):
                current = fp32_currents(counts, decay, unit_additions)
                recovered, diagnostics = subject._decode(
                    current, decay.astype(np.float64)
                )
                np.testing.assert_array_equal(recovered, counts)
                self.assertEqual(recovered.dtype, np.dtype(np.int64))
                self.assertGreater(diagnostics["max_interval_width"], 0.0)

    def test_rejects_fractional_event_instead_of_rounding(self):
        current = np.asarray([[0.0], [0.5]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "noninteger or ambiguous"):
            subject._decode(current, np.asarray([1.0]))

    def test_rejects_interval_containing_multiple_nonnegative_integers(self):
        # Point reconstruction is exactly zero; its rounding interval is broad.
        current = np.asarray([[4096.0], [4096.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "noninteger or ambiguous"):
            subject._decode(current, np.asarray([1.0]))


class ConfigurationTests(unittest.TestCase):
    def test_preserves_explicit_support_without_deriving_it_from_window(self):
        cfg = base_config()
        cfg["finite_window"] = {"start_ms": 10.0, "stop_ms": 20.0}
        cfg["expected_arrival_support"] = {"first_tick": 104, "last_tick": 204}
        result = subject._config(cfg)
        self.assertEqual(result["expected_arrival_support"], cfg["expected_arrival_support"])
        self.assertEqual(result["finite_window"], cfg["finite_window"])
        self.assertNotIn("expected_arrival_support", subject._config(base_config()))

    def test_rejects_nonpositive_or_nonfinite_times(self):
        for key in ("dt_ms", "tau_syn_ms", "delay_ms", "duration_ms"):
            for value in (0.0, -0.1, math.nan, math.inf):
                with self.subTest(key=key, value=value):
                    cfg = base_config()
                    cfg[key] = value
                    with self.assertRaises(ValueError):
                        subject._config(cfg)

    def test_rejects_invalid_finite_windows(self):
        for start, stop in ((-1.0, 2.0), (2.0, 1.0), (math.nan, 2.0), (1.0, math.inf)):
            with self.subTest(start=start, stop=stop):
                cfg = base_config()
                cfg["finite_window"] = {"start_ms": start, "stop_ms": stop}
                with self.assertRaises(ValueError):
                    subject._config(cfg)

    def test_rejects_invalid_support_ticks(self):
        for first, last in ((-1, 2), (3, 2), (1.5, 3), (True, 3), (0, 2.5)):
            with self.subTest(first=first, last=last):
                cfg = base_config()
                cfg["expected_arrival_support"] = {"first_tick": first, "last_tick": last}
                with self.assertRaises(ValueError):
                    subject._config(cfg)

    def test_accepts_fp32_clock_roundoff_across_full_duration(self):
        expected = np.arange(20001, dtype=np.int64)
        times = np.float32(expected.astype(np.float64) * float(np.float32(0.1)))
        np.testing.assert_array_equal(
            subject._clock_ticks(times.astype(np.float64), 0.1), expected
        )

    def test_rejects_shifted_duplicate_missing_or_nonfinite_times(self):
        for times in (
            [0.0, 0.1, 0.25], [0.0, 0.1, 0.1], [0.0, 0.2, 0.1],
            [0.0, 0.1, 0.3], [0.0, math.nan], [0.0, math.inf],
        ):
            with self.subTest(times=times):
                with self.assertRaises(ValueError):
                    subject._clock_ticks(np.asarray(times), 0.1)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="poisson-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name, counts, means, groups=None, support="all"):
        counts = np.asarray(counts, dtype=np.int64)
        means = np.asarray(means, dtype=np.float64)
        groups = np.zeros(counts.shape[1], dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
        output = self.root / name
        output.mkdir()
        cfg = base_config()
        cfg["duration_ms"] = len(counts) * cfg["dt_ms"]
        unique_groups = sorted(set(int(value) for value in groups))
        cfg["rates_hz"] = [
            float(means[np.flatnonzero(groups == group)[0]]) * 1000 / cfg["dt_ms"]
            for group in unique_groups
        ]
        cfg["targets_per_rate"] = counts.shape[1] // len(unique_groups)
        if support == "all":
            cfg["expected_arrival_support"] = {"first_tick": 0, "last_tick": len(counts) - 1}
        elif support is not None:
            cfg["expected_arrival_support"] = support
        (output / "acquisition.json").write_text(json.dumps({
            "status": "EXECUTED", "config": cfg,
            "fixture": "synthetic CPU analysis control, not a native receipt",
        }))
        np.savez_compressed(
            output / "counts.npz", counts=counts,
            arrival_ticks=np.arange(len(counts), dtype=np.int64),
            lambda_per_target=means, generator_index=groups,
        )
        return output, cfg

    @staticmethod
    def of_kind(report, kind):
        return [test for test in report["tests"] if test["kind"] == kind]

    def test_zero_rate_positive_control_and_exact_interval_boundaries(self):
        output, _ = self.fixture("zero-rate", np.zeros((32, 2)), [0.0, 0.0], groups=[0, 1])
        report = subject.analyze(output)
        self.assertEqual(report["law_test_outcome"], "NOT_REJECTED")
        self.assertFalse(report["scientific_reference_qualified"])
        self.assertEqual(report["family_alpha"], 0.05)
        self.assertEqual(report["per_test_alpha"], 0.05 / len(report["tests"]))
        self.assertTrue(all(test["p_value"] == 1.0 for test in report["tests"]))
        self.assertTrue(all(row["count_mean"] == row["count_variance"] == 0.0 for row in report["target_moments"]))
        alpha = report["per_test_alpha"]
        expected_upper = -math.expm1(math.log(alpha / 2) / 32)
        interval = self.of_kind(report, "multiplicity")[0]["exact_binomial_ci"]
        self.assertEqual(interval[0], 0.0)
        self.assertAlmostEqual(interval[1], expected_upper, places=10)
        zero_interval = self.of_kind(report, "zero_count")[0]["exact_binomial_ci"]
        self.assertAlmostEqual(zero_interval[0], math.exp(math.log(alpha / 2) / 32), places=10)
        self.assertEqual(zero_interval[1], 1.0)

    def test_rate_controls_reject_both_tails(self):
        for value in (0, 4):
            with self.subTest(value=value):
                output, _ = self.fixture("rate-" + str(value), np.full((128, 1), value), [1.0])
                report = subject.analyze(output)
                test = self.of_kind(report, "poisson_total")[0]
                self.assertEqual(report["law_test_outcome"], "REJECTED")
                self.assertTrue(test["reject"])
                if value == 0:
                    self.assertTrue(math.isclose(test["p_value"], 2 * math.exp(-128), rel_tol=2e-12, abs_tol=0.0))

    def test_suppressed_multiplicity_rejected_even_with_correct_total(self):
        output, _ = self.fixture("capped", np.ones((512, 1)), [1.0])
        report = subject.analyze(output)
        self.assertFalse(self.of_kind(report, "poisson_total")[0]["reject"])
        test = self.of_kind(report, "multiplicity")[0]
        self.assertEqual(test["successes"], 0)
        self.assertTrue(test["reject"])

    def test_shared_streams_rejected_within_and_across_generators(self):
        pattern = np.tile([0, 1, 0, 2, 1, 3, 0, 1], 128)
        counts = np.repeat(pattern[:, None], 4, axis=1)
        output, _ = self.fixture("shared", counts, np.ones(4), groups=[0, 0, 1, 1])
        report = subject.analyze(output)
        for kind in ("same_generator_zero_disagreement", "cross_generator_zero_disagreement"):
            tests = self.of_kind(report, kind)
            self.assertTrue(tests)
            self.assertTrue(all(test["successes"] == 0 and test["reject"] for test in tests))

    def test_serially_duplicated_pairs_rejected(self):
        pattern = np.tile([0, 1, 0, 2, 1, 3, 0, 1], 64)
        counts = np.repeat(pattern, 2)[:, None]
        output, _ = self.fixture("serial", counts, [1.0])
        report = subject.analyze(output)
        test = self.of_kind(report, "serial_zero_disagreement")[0]
        self.assertEqual(test["trials"], len(counts) // 2)
        self.assertEqual(test["successes"], 0)
        self.assertTrue(test["reject"])

    def test_outside_support_event_is_not_trimmed_into_nonrejection(self):
        counts = np.zeros((32, 1), dtype=np.int64)
        counts[0, 0] = 1
        output, _ = self.fixture(
            "outside", counts, [0.0],
            support={"first_tick": 1, "last_tick": 30},
        )
        report = subject.analyze(output)
        self.assertEqual(report["rejection_count"], 0)
        self.assertTrue(report["support_violation"])
        self.assertEqual(report["outside_support"]["event_count"], 1)
        self.assertEqual(report["law_test_outcome"], "REJECTED")

    def test_missing_support_remains_not_tested(self):
        output, _ = self.fixture("unspecified", np.ones((32, 1)), [1.0], support=None)
        report = subject.analyze(output / "counts.npz")
        self.assertEqual(report["law_test_outcome"], "NOT_TESTED")
        self.assertEqual(report["tests"], [])
        self.assertIsNone(report["outside_support"])
        self.assertFalse(report["scientific_reference_qualified"])

    def run_cli(self, output, cfg):
        config = output / "protocol.yaml"
        # JSON is a valid YAML mapping and keeps this fixture dependency-free.
        config.write_text(json.dumps(cfg))
        return subprocess.run(
            [
                sys.executable, "-m", "Libs.population_dynamics.nest_gpu_poisson",
                "--mode", "analyze", "--config", str(config), "--output", str(output),
            ],
            cwd=Path(subject.__file__).resolve().parents[2],
            env=os.environ.copy(), capture_output=True, text=True, check=False,
        )

    def test_real_analysis_cli_exit_codes(self):
        for name, mean, support, expected_code, expected_outcome in (
            ("cli-pass", 0.0, "all", 0, "NOT_REJECTED"),
            ("cli-reject", 1.0, "all", 1, "REJECTED"),
            ("cli-untested", 1.0, None, 2, "NOT_TESTED"),
        ):
            with self.subTest(name=name):
                output, cfg = self.fixture(name, np.zeros((128, 1)), [mean], support=support)
                run = self.run_cli(output, cfg)
                self.assertEqual(run.returncode, expected_code, run.stdout + run.stderr)
                self.assertEqual(json.loads(run.stdout)["law_test_outcome"], expected_outcome)

    def test_cli_refuses_config_mismatch_before_analysis_write(self):
        output, cfg = self.fixture("cli-mismatch", np.zeros((32, 1)), [0.0])
        cfg["seed"] += 1
        run = self.run_cli(output, cfg)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("analysis configuration differs", run.stderr)
        self.assertFalse((output / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()

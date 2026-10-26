"""Deterministic source-count precision fixtures; no source simulation."""
import copy
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from . import pfc_source_history_precision as precision

HEADER = ("circuit_index", "block_index", "episode_id", "condition_id", "phase_index",
          "start_tick", "stop_tick", "start_ms", "stop_ms", "excitatory_spikes", "inhibitory_spikes",
          "excitatory_rate_hz", "inhibitory_rate_hz", "all_unit_rate_hz")


class SourceHistoryPrecisionTests(unittest.TestCase):
    def counts(self, circuit, block, condition, phase):
        pair = condition % 5
        cue = condition < 5
        if circuit == 23 and pair == 0:
            return 0, 0
        if circuit == 23 and pair == 1:
            return (10, 7) if cue else (20, 9)
        step = (-5, -2, 0, 2, 4, 8)[block - 3]
        excitatory = 100 + 10 * phase + pair
        inhibitory = 80 + 5 * phase + pair
        return (excitatory + step * (pair + 1), inhibitory + step) if cue else (excitatory, inhibitory)

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def binding(self, path):
        data = path.read_bytes()
        return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def fixture(self, root, omit=None):
        rows = []
        for circuit in precision.CIRCUITS:
            for block in precision.BLOCKS:
                for condition in precision.CONDITIONS:
                    if omit == (circuit, block, condition):
                        continue
                    name = precision.episode_id(circuit, block, condition)
                    circuit_id = f"pfc_working_memory_circuit_{circuit:03d}"
                    identity = dict(circuit_index=circuit, block_index=block, condition_index=condition,
                                    episode_id=name, circuit_id=circuit_id,
                                    block_id=circuit_id + f"_block_{block:03d}",
                                    initial_state_seed=100000 + circuit * 100 + block,
                                    background_seed=200000 + circuit * 100 + block)
                    condition_id = f"source_condition_{condition:02d}"
                    phase_rows = []
                    for phase in range(8):
                        start, stop = phase * 50000, (phase + 1) * 50000
                        excitatory, inhibitory = self.counts(circuit, block, condition, phase)
                        phase_rows.append(dict(circuit_index=circuit, block_index=block, episode_id=name,
                            condition_id=condition_id, phase_index=phase, start_tick=start, stop_tick=stop,
                            start_ms=str(Decimal(start) / 80), stop_ms=str(Decimal(stop) / 80),
                            excitatory_spikes=excitatory, inhibitory_spikes=inhibitory,
                            excitatory_rate_hz="0.0", inhibitory_rate_hz="0.0", all_unit_rate_hz="0.0"))
                    path = root / (name + ".csv")
                    self.write_csv(path, phase_rows)
                    rows.append(dict(identity=identity, condition_id=condition_id,
                        config_sha256=hashlib.sha256(name.encode()).hexdigest(),
                        source_signature=dict(clock="fine", circuit=circuit, condition=condition),
                        phase_csv=self.binding(path)))
        manifest = dict(schema=precision.SCHEMA, design_circuits=list(precision.CIRCUITS),
                        block_indices=list(precision.BLOCKS), condition_indices=list(precision.CONDITIONS),
                        dt_ms=.0125, duration_ticks=400000, core_units=100000,
                        population_sizes={"E": 80000, "I": 20000}, episodes=rows)
        path = root / "manifest.json"
        self.write_manifest(path, manifest)
        return path, manifest

    def write_manifest(self, path, value):
        path.write_bytes(precision.canonical_json(value))

    def read_rows(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_n6_and_n3_cohorts_match_explicit_rate_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self.fixture(root)
            output = root / "output"
            report = precision.run(manifest_path, output)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual((report["actual_episodes"], report["required_episodes"]), (240, 240))
            self.assertEqual(report["paired_history_phase_rate_rows"], 2880)
            rows = self.read_rows(output / "history_precision_summary.csv")
            for cohort, blocks in precision.COHORTS.items():
                for population, units in precision.POPULATIONS.items():
                    with self.subTest(cohort=cohort, population=population):
                        row = next(row for row in rows if row["circuit_index"] == "63"
                                   and row["cohort"] == cohort and row["cue_condition_index"] == "2"
                                   and row["phase_index"] == "4" and row["population"] == population)
                        differences = []
                        for block in blocks:
                            cue = self.counts(63, block, 2, 4)
                            omission = self.counts(63, block, 7, 4)
                            if population == "E": difference = cue[0] - omission[0]
                            elif population == "I": difference = cue[1] - omission[1]
                            else: difference = sum(cue) - sum(omission)
                            differences.append(difference)
                        rates = [value * 80000 / (units * 50000) for value in differences]
                        expected_variance = statistics.variance(rates)
                        self.assertEqual(int(row["n_histories"]), len(blocks))
                        self.assertTrue(math.isclose(float(row["mean_rate_difference_hz"]),
                                                    statistics.mean(rates), rel_tol=1e-12, abs_tol=1e-25))
                        self.assertTrue(math.isclose(float(row["sample_variance_hz2"]), expected_variance,
                                                    rel_tol=1e-12, abs_tol=1e-25))
                        self.assertTrue(math.isclose(float(row["estimated_sem_hz"]),
                                                    math.sqrt(expected_variance / len(blocks)),
                                                    rel_tol=1e-12, abs_tol=1e-25))
                        n = len(blocks)
                        self.assertEqual(int(row["count_variance_numerator"]),
                                         n * sum(value * value for value in differences) - sum(differences) ** 2)
                        self.assertEqual(int(row["count_variance_denominator"]), n * (n - 1))
            self.assertFalse(report["uncertainty_coverage_qualified"])
            self.assertFalse(report["source_qualification_granted"])

    def test_quiet_zero_variance_and_negative_contrasts_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self.fixture(root)
            output = root / "output"
            precision.run(manifest_path, output)
            summaries = self.read_rows(output / "history_precision_summary.csv")
            for row in summaries:
                if row["circuit_index"] == "23" and row["cue_condition_index"] in ("0", "1"):
                    self.assertEqual(int(row["count_variance_numerator"]), 0)
                    self.assertEqual(float(row["sample_variance_hz2"]), 0)
                    self.assertEqual(float(row["estimated_sem_hz"]), 0)
                    if row["cue_condition_index"] == "0":
                        self.assertEqual(float(row["mean_rate_difference_hz"]), 0)
                    else:
                        self.assertLess(float(row["mean_rate_difference_hz"]), 0)
            paired = self.read_rows(output / "paired_history_phase_rates.csv")
            negative = next(row for row in paired if row["circuit_index"] == "23"
                            and row["cue_condition_index"] == "1" and row["population"] == "E")
            self.assertEqual(int(negative["count_difference"]), -10)

    def test_missing_episode_keeps_fixed_denominators_and_withholds_pooled_circuit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = 23, 8, 9
            manifest_path, _ = self.fixture(root, omit=missing)
            output = root / "output"
            report = precision.run(manifest_path, output)
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["actual_episodes"], 239)
            self.assertEqual(report["required_episodes"], 240)
            self.assertEqual(report["missing_episode_ids"], [precision.episode_id(*missing)])
            rows = self.read_rows(output / "history_precision_summary.csv")
            circuit_rows = [row for row in rows if row["circuit_index"] == "23"]
            self.assertEqual({row["cohort"] for row in circuit_rows}, {"pilot_fine"})
            self.assertEqual({row["n_histories"] for row in circuit_rows}, {"3"})
            self.assertEqual(report["design_circuits"], [23, 63, 52, 86])

    def test_duplicate_stream_and_source_signature_rejections(self):
        for kind in ("duplicate", "stream", "reused_history", "signature", "coarse", "validation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = self.fixture(root)
                if kind == "duplicate": manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
                elif kind == "stream": manifest["episodes"][1]["identity"]["initial_state_seed"] += 1000
                elif kind == "reused_history":
                    old = manifest["episodes"][0]["identity"]["background_seed"]
                    for episode in manifest["episodes"]:
                        if episode["identity"]["circuit_index"] == 23 and episode["identity"]["block_index"] == 8:
                            episode["identity"]["background_seed"] = old
                elif kind == "signature": manifest["episodes"][50]["source_signature"]["clock"] = "coarse"
                elif kind == "coarse": manifest["dt_ms"] = .05
                else: manifest["design_circuits"][0] = 74
                self.write_manifest(manifest_path, manifest)
                output = root / "output"
                with self.assertRaises(ValueError):
                    precision.run(manifest_path, output)
                self.assertFalse(output.exists())

    def test_phase_counts_boundaries_and_file_hash_are_checked(self):
        for kind in ("negative_count", "fractional_count", "boundary", "native_ms", "hash"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = self.fixture(root)
                episode = manifest["episodes"][0]
                phase_path = Path(episode["phase_csv"]["path"])
                rows = self.read_rows(phase_path)
                if kind == "negative_count": rows[0]["excitatory_spikes"] = "-1"
                elif kind == "fractional_count": rows[0]["inhibitory_spikes"] = "1.5"
                elif kind == "boundary":
                    rows[1]["start_tick"] = "50001"
                    rows[1]["start_ms"] = str(Decimal(50001) / 80)
                elif kind == "native_ms": rows[1]["start_ms"] = "626"
                else: rows[0]["excitatory_spikes"] = "1"
                self.write_csv(phase_path, rows)
                if kind != "hash": episode["phase_csv"] = self.binding(phase_path)
                self.write_manifest(manifest_path, manifest)
                output = root / "output"
                with self.assertRaises(ValueError):
                    precision.run(manifest_path, output)
                self.assertFalse(output.exists())

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self.fixture(root)
            output = root / "output"
            precision.run(manifest_path, output)
            report = (output / "report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "IMMUTABLE_OUTPUT_EXISTS"):
                precision.run(manifest_path, output)
            self.assertEqual((output / "report.json").read_bytes(), report)


if __name__ == "__main__":
    unittest.main()

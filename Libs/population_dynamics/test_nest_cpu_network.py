"""Focused CPU/native tests for original-source count-driven replay."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Libs.population_dynamics import nest_cpu_network as subject
from Libs.population_dynamics.network_snapshot import NEURON_PARAMETERS, SCHEMA
from Libs.population_dynamics.nest_reference import EDGE_DTYPE


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def refresh_manifest(snapshot):
    names = ("edges.npy", "unit_ids.npy", "populations.npy", "positions_mm.npy", "cortical_parameters.npy", "metadata.json")
    write_json(snapshot / "manifest.json", {
        "schema": SCHEMA,
        "artifacts": {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in names},
    })


def fixture(root):
    snapshot, drive = root / "snapshot", root / "drive"
    snapshot.mkdir()
    drive.mkdir()
    ids = np.asarray([11, 23, 42], dtype=np.int64)
    edges = np.asarray([
        (11, 100, 2.0, 0.2), (23, 100, 4.0, 0.2),
        (23, 11, 0.125, 0.3), (23, 11, 0.25, 0.3),
        (11, 23, -0.125, 0.3), (42, 200, 1.0, 0.2),
        (11, 42, 1.5, 0.2),
    ], dtype=EDGE_DTYPE)
    edges["delay"] = np.rint(edges["delay"] / 0.1) * 0.1
    edges.sort(order=list(EDGE_DTYPE.names))
    parameter = np.asarray([
        [250., 10., 0.5, 0.5, -65.123456789, 1000.23456789, -65.34567891, 2., 0., -64.987654321],
        [250., 10., 0.5, 0.5, -65.223456789, 1000.23456789, -65.34567891, 2., 0., -64.887654321],
    ])
    # Match native export: absolute voltage is reconstructed from relative state.
    for column in (5, 6, 9):
        parameter[:, column] = parameter[:, 4] + (parameter[:, column] - parameter[:, 4])
    for name, value in {
        "unit_ids": ids, "populations": np.asarray(["L23E", "L23I", "TC"]),
        "positions_mm": np.asarray([[0., 0.], [0.1, 0.], [0., 0.1]]),
        "cortical_parameters": parameter, "edges": edges,
    }.items():
        np.save(snapshot / (name + ".npy"), value, allow_pickle=False)
    metadata = {
        "schema": SCHEMA, "source_revision": subject.SOURCE_REVISION,
        "native_time_ms": 0., "dt_ms": 0.1, "duration_ms": 5.,
        "cortical_neuron_model": "iaf_psc_exp", "thalamic_neuron_model": "parrot_neuron",
        "parameter_columns": list(NEURON_PARAMETERS),
        "devices": [
            {"id": 100, "model": "poisson_generator", "parameters": {"rate": 128., "origin": 0., "start": 0., "stop": 1e30}},
            {"id": 200, "model": "spike_generator", "parameters": {"initial_spike_times_ms": [2., 3.], "effective_spike_times_ms": [2., 3.2]}},
        ],
        "connectivity": {"connection_count": len(edges)},
        "config": {
            "random_streams": {"background_drive": 55004},
            "pulse_control": {"decision_time_ms": 2.5, "initial_emission_ms": [2., 3.], "future_emission_ms": [3.2]},
        },
    }
    write_json(snapshot / "metadata.json", metadata)
    refresh_manifest(snapshot)
    counts = np.zeros((50, 2), dtype=np.int64)
    counts[5] = [3, 5]
    counts[10] = [2, 1]
    counts[49] = [1, 0]
    write_drive(drive, snapshot, counts)
    return snapshot, drive


def write_drive(drive, snapshot, counts, **changes):
    np.savez_compressed(drive / "counts.npz", unit_ids=np.asarray([11, 23]),
                        arrival_ticks=np.arange(50), counts=counts, dt_ms=0.1)
    receipt = {
        "schema": subject.DRIVE_SCHEMA,
        "snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
        "counts_sha256": hashlib.sha256((drive / "counts.npz").read_bytes()).hexdigest(),
        "dt_ms": 0.1, "simulation_duration_ms": 5.0,
        "last_complete_arrival_tick": 49, "counts_decoding_validated": True,
    }
    receipt.update(changes)
    write_json(drive / "drive.json", receipt)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-cpu-plan-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot, self.drive = fixture(self.root)

    def test_positive_counts_become_delayed_emissions_with_multiplicity(self):
        plan = subject.prepare(self.snapshot, self.drive)
        program = plan["background_program"]
        np.testing.assert_array_equal(program[0]["emission_ticks"], [3, 8, 47])
        self.assertEqual(program[0]["multiplicities"], [3, 2, 1])
        np.testing.assert_array_equal(program[1]["emission_ticks"], [3, 8])
        self.assertEqual(program[1]["multiplicities"], [5, 1])
        self.assertEqual(len(plan["expected_edges"]), 7)
        self.assertEqual(len(plan["thalamic_ids"]), 1)

    def test_gpu_quantization_uses_relative_voltage(self):
        plan = subject.prepare(self.snapshot, self.drive, quantization="gpu_fp32_inputs")
        original = dict(zip(NEURON_PARAMETERS, np.load(self.snapshot / "cortical_parameters.npy").T))
        for name in ("V_m", "V_th", "V_reset"):
            expected = original["E_L"].astype(np.float32).astype(float) + (original[name] - original["E_L"]).astype(np.float32).astype(float)
            np.testing.assert_array_equal(plan["parameters"][name], expected)
        self.assertFalse(np.array_equal(plan["parameters"]["V_m"], original["V_m"].astype(np.float32).astype(float)))

    def test_rejects_counts_hash_change(self):
        with (self.drive / "counts.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            subject.prepare(self.snapshot, self.drive)

    def test_rejects_unrepresentable_preinitial_or_zero_time_emissions(self):
        for arrival in (1, 2):
            with self.subTest(arrival=arrival):
                counts = np.zeros((50, 2), dtype=np.int64)
                counts[arrival, 0] = 1
                write_drive(self.drive, self.snapshot, counts)
                with self.assertRaisesRegex(ValueError, "prior or zero-time"):
                    subject.prepare(self.snapshot, self.drive)

    def test_rejects_invalid_drive_metadata(self):
        original = json.loads((self.drive / "drive.json").read_text())
        for change in (
            {"schema": "unknown"}, {"counts_decoding_validated": False},
            {"snapshot_manifest_sha256": "0" * 64}, {"dt_ms": 0.001},
            {"simulation_duration_ms": 4.0}, {"last_complete_arrival_tick": 48},
        ):
            with self.subTest(change=change):
                bad = dict(original)
                bad.update(change)
                write_json(self.drive / "drive.json", bad)
                with self.assertRaises(ValueError):
                    subject.prepare(self.snapshot, self.drive)

    def test_rejects_count_identity_grid_and_numeric_errors(self):
        with np.load(self.drive / "counts.npz") as original:
            clean = {key: original[key].copy() for key in original.files}
        variants = [
            {"unit_ids": np.asarray([23, 11])},
            {"arrival_ticks": np.arange(1, 51)},
            {"counts": np.zeros((50, 2), dtype=float)},
            {"counts": np.full((50, 2), -1, dtype=np.int64)},
            {"dt_ms": np.asarray(0.001)},
        ]
        receipt = json.loads((self.drive / "drive.json").read_text())
        for change in variants:
            with self.subTest(change=list(change)):
                data = dict(clean)
                data.update(change)
                np.savez_compressed(self.drive / "counts.npz", **data)
                current = dict(receipt, counts_sha256=hashlib.sha256((self.drive / "counts.npz").read_bytes()).hexdigest())
                write_json(self.drive / "drive.json", current)
                with self.assertRaises(ValueError):
                    subject.prepare(self.snapshot, self.drive)

    def test_rejects_changed_pulse_prefix_even_with_updated_manifest(self):
        metadata = json.loads((self.snapshot / "metadata.json").read_text())
        metadata["devices"][1]["parameters"]["effective_spike_times_ms"] = [2.1, 3.2]
        write_json(self.snapshot / "metadata.json", metadata)
        refresh_manifest(self.snapshot)
        receipt = json.loads((self.drive / "drive.json").read_text())
        receipt["snapshot_manifest_sha256"] = hashlib.sha256((self.snapshot / "manifest.json").read_bytes()).hexdigest()
        write_json(self.drive / "drive.json", receipt)
        with self.assertRaisesRegex(ValueError, "changed the source history"):
            subject.prepare(self.snapshot, self.drive)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Kept under the caller's task scratch so reports survive failed tests.
        cls.root = Path(tempfile.mkdtemp(prefix="native-cpu-execution-", dir=os.environ.get("TMPDIR")))
        cls.snapshot, cls.drive = fixture(cls.root)
        cls.outputs = {}
        for name, quantization in (("source", "source_float64"), ("repeat", "source_float64"), ("gpu_inputs", "gpu_fp32_inputs")):
            output = cls.root / name
            if name == "repeat":
                command = [sys.executable, "-m", "Libs.population_dynamics.nest_cpu_network",
                           "--snapshot", str(cls.snapshot), "--drive", str(cls.drive),
                           "--output", str(output), "--schedule", "effective",
                           "--quantization", quantization]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                (cls.root / "repeat_cli.log").write_text(completed.stdout + completed.stderr)
                rc = completed.returncode
            else:
                rc = subject.run(cls.snapshot, cls.drive, output, "effective", quantization)
            cls.outputs[name] = output
            if rc:
                raise AssertionError((output / "report.json").read_text())
        print("NATIVE_TEST_ARTIFACTS", cls.root)

    def test_original_edges_and_multiplicity_arrivals(self):
        output = self.outputs["source"]
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["nest_version"], "3.10.0")
        self.assertTrue(report["checks"]["all_source_edges_exact"])
        self.assertEqual(report["source_connections"], 7)
        self.assertEqual(report["background_count_total"], 12)
        with np.load(output / "states.npz") as archive:
            row = np.flatnonzero(archive["ticks"] == 5)[0]
            ex = np.flatnonzero(archive["fields"] == "I_syn_ex")[0]
            np.testing.assert_array_equal(archive["states"][row, :, ex], [6.0, 20.0])
            self.assertEqual(archive["recording_phase"].item(), "CPU post-arrival PSCs")

    def test_original_parrot_and_pulse_edges_preserve_TC_timing(self):
        with np.load(self.outputs["source"] / "spikes.npz") as archive:
            mask = archive["spike_unit_ids"] == 42
            np.testing.assert_array_equal(archive["spike_ticks"][mask], [22, 34])

    def test_native_fresh_process_replay_is_exact(self):
        for name in ("spikes.npz", "states.npz"):
            with np.load(self.outputs["source"] / name) as first, np.load(self.outputs["repeat"] / name) as second:
                self.assertEqual(set(first.files), set(second.files))
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])

    def test_quantized_control_records_exact_GPU_inputs(self):
        output = self.outputs["gpu_inputs"]
        plan = subject.prepare(self.snapshot, self.drive, quantization="gpu_fp32_inputs")
        readback = json.loads((output / "parameter_readback.json").read_text())
        for key, values in plan["parameters"].items():
            np.testing.assert_array_equal(readback[key], values)
        report = json.loads((output / "report.json").read_text())
        self.assertFalse(report["scientific_reference_qualified"])
        self.assertIn("Integer multiplicities are not modified", report["accumulation_caveat"])


if __name__ == "__main__":
    unittest.main()

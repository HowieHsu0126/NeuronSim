"""Checkpoint publication boundaries and terminal failure propagation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .population_residual_observation_execution import CheckpointPublisher, execute


class Publisher:
    def __init__(self):
        self.calls = []
        self.fail = False

    def hdfs(self, *arguments):
        self.calls.append(("hdfs", arguments))

    def persist(self, work, cell, prefix, exit_code):
        if self.fail:
            raise RuntimeError("publication unavailable")
        self.calls.append(("persist", prefix, exit_code))
        return dict(checkpoint_path=prefix + "/readout_diagnostic",
                    persisted=True, exit_code=exit_code, result_bytes=1,
                    result_sha256="fixture", chunks=[])


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.cell = self.output / "qualify/seed_41/current"
        self.cell.mkdir(parents=True)
        (self.cell / "reference.npz").write_bytes(b"fixture")
        (self.cell / "reference.json").write_text("{}")
        self.path = self.cell / "epoch_checkpoint.pt"
        self.publisher = Publisher()
        self.callback = CheckpointPublisher(self.publisher, self.root, self.output, "hdfs://fixture")
        self.state = dict(epoch=0, selected_epoch=0, status="PARTIAL", best_loss=1.,
                          identity=dict(binding=dict(phase="qualify", seed=41, mode="current")))
        self.save()
        environment = patch.dict("os.environ", RH2_JOB_RUN_ID="fixture_run",
                                 ARNOLD_TRIAL_ID="fixture_trial", RH2_CHECKPOINT_HDFS="hdfs://fixture")
        gpu = patch("torch.cuda.is_available", return_value=False)
        environment.start()
        gpu.start()
        self.addCleanup(environment.stop)
        self.addCleanup(gpu.stop)

    def save(self):
        torch.save(self.state, self.path)

    def test_first_periodic_and_forced_export_have_distinct_receipts(self):
        with patch("time.monotonic", return_value=0):
            self.callback(self.cell, self.path, 0)
        self.state["epoch"] = 1
        self.save()
        with patch("time.monotonic", return_value=599):
            self.callback(self.cell, self.path, 1)
        self.assertEqual(len(self.callback.records), 1)
        with patch("time.monotonic", return_value=600):
            self.callback(self.cell, self.path, 1)
        (self.cell / "fit.json").write_text("{}")
        self.callback(self.cell, self.path, 1, True)
        self.callback(self.cell, self.path, 1, True)
        self.assertEqual(len(self.callback.records), 3)
        self.assertEqual(len({row["receipt_uri"] for row in self.callback.records}), 3)
        self.assertFalse(self.callback.records[0]["fit_export_available"])
        self.assertTrue(self.callback.records[-1]["fit_export_available"])
        self.assertTrue(all(row["source_fit_process_exit_code"] is None for row in self.callback.records))

    def test_publication_failure_does_not_claim_persistence(self):
        self.publisher.fail = True
        original = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "publication unavailable"):
            self.callback(self.cell, self.path, 0)
        self.assertEqual(self.callback.records, [])
        self.assertEqual(self.path.read_bytes(), original)

    def test_epoch_mismatch_and_missing_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch differs"):
            self.callback(self.cell, self.path, 1)
        (self.cell / "reference.json").unlink()
        with self.assertRaisesRegex(ValueError, "reference"):
            self.callback(self.cell, self.path, 0)
        self.assertEqual(self.publisher.calls, [])

    def test_runner_retains_failure_and_nonzero_terminal_code(self):
        args = SimpleNamespace(output=self.output, inputs=self.root, manifest=self.root,
            config=self.root, phase="qualify", seeds=[41], modes=["current"],
            device="cpu", epoch_limit=1)
        with patch("population_dynamics.population_residual_observation_execution.fit",
                   side_effect=ValueError("fitting failure")):
            self.assertEqual(execute(args, self.publisher), 1)
        report = json.loads((self.output / "execution.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("fitting failure", report["error"])
        self.assertEqual(self.publisher.calls[-1][-1], 1)

    def test_clean_bounded_exit_remains_partial(self):
        args = SimpleNamespace(output=self.output, inputs=self.root, manifest=self.root,
            config=self.root, phase="qualify", seeds=[41], modes=["current"],
            device="cpu", epoch_limit=1)
        with patch("population_dynamics.population_residual_observation_execution.fit",
                   return_value={"complete": False}) as fitter:
            self.assertEqual(execute(args, self.publisher), 0)
        self.assertIsInstance(fitter.call_args.kwargs["checkpoint_callback"], CheckpointPublisher)
        report = json.loads((self.output / "execution.json").read_text())
        self.assertEqual(report["status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()

def test_native_epoch_checkpoint_publishes_actual_optimizer_state(tmp_path, monkeypatch):
    from .population_residual_observation_training import run_epochs
    from .population_residual_observation_fitting import save_predictor
    from .test_population_residual_observation import arguments
    from .test_population_residual_memory import predictor
    monkeypatch.setenv("RH2_JOB_RUN_ID", "fixture_run")
    monkeypatch.setenv("ARNOLD_TRIAL_ID", "fixture_trial")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "output"
    cell = output / "qualify/seed_41/history"
    cell.mkdir(parents=True)
    batch = arguments(mode="history", seed=41, dtype=torch.float32)
    model = batch.pop("model")
    reference = predictor("history", memory_gain=.15, constant=.01)
    save_predictor(cell / "reference.npz", reference)
    (cell / "reference.json").write_text("{}")
    callback = CheckpointPublisher(Publisher(), tmp_path, output, "hdfs://fixture")
    config = dict(learning_rate=.0001, regularization_weight=.0001, max_epochs=2,
                  patience=2, prediction_particles=8, unit_chunk_size=2)
    result = run_epochs(model, reference, lambda case: batch, ["fit"], ["validation"],
        config, "qualify", 41, cell, {"phase": "qualify", "seed": 41, "mode": "history"},
        epoch_limit=1, checkpoint_callback=lambda path, epoch: callback(cell, path, epoch))
    callback(cell, cell / "epoch_checkpoint.pt", 1, True)
    checkpoint = torch.load(cell / "epoch_checkpoint.pt", weights_only=True)
    assert result["status"] == "PARTIAL"
    assert len(checkpoint["optimizer"]["state"]) == 2
    assert set(checkpoint["rng"]) == {"order", "train_initial", "train_dynamics"}
    assert [row["completed_epochs"] for row in callback.records] == [0, 1]
    assert all(row["checkpoint_status"] == "PARTIAL" for row in callback.records)

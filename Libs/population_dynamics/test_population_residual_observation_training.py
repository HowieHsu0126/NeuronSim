"""Selection, failure and exact-resume checks for the native epoch optimizer."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from population_dynamics import population_residual_observation_training as training


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.correction_weight = torch.nn.Parameter(torch.tensor([0.2]))
        self.correction_bias = torch.nn.Parameter(torch.tensor([0.1]))


class EpochTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(learning_rate=.01, regularization_weight=0.,
            max_epochs=3, patience=2, prediction_particles=8, unit_chunk_size=4096)
        self.reference = SimpleNamespace(alpha=1., **{
            key: torch.ones(1) for key in training._REF_ARRAYS})
        self.tie = False
        self.calls = []
        for item in (
            patch.object(training, "residual_observation_mixture_loss", side_effect=self.loss),
            patch.object(training, "standardized_ridge_penalty",
                         side_effect=lambda model, reference: model.correction_weight.square().sum()),
        ):
            item.start()
            self.addCleanup(item.stop)

    def loss(self, model, target, case, initial_generator, dynamics_generator, **kwargs):
        noise = .02 * (torch.randn((), generator=initial_generator)
                       + torch.randn((), generator=dynamics_generator))
        value = (model.correction_weight + model.correction_bias + noise - target).square().sum()
        if self.tie and case == "v":
            value = value * 0 + 1
        return dict(loss=value, native_nll_h4=value, particle_ess_h4=torch.tensor(1.))

    def loader(self, case):
        self.calls.append(case)
        return dict(target=torch.tensor(2.), case=case)

    def run_fit(self, model, root, seed=41, **kwargs):
        return training.run_epochs(model, self.reference, self.loader, ["a", "b"], ["v"],
            self.config, "qualify", seed, root, {"input": "same"}, **kwargs)

    def test_resume_matches_uninterrupted_all_three_seeds(self):
        for seed in (41, 42, 43):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as root:
                a, b = Model(), Model()
                full = self.run_fit(a, Path(root) / "full", seed)
                partial = self.run_fit(b, Path(root) / "resumed", seed, epoch_limit=1)
                self.assertEqual(partial["status"], "PARTIAL")
                resumed = self.run_fit(b, Path(root) / "resumed", seed)
                self.assertEqual(full["records"], resumed["records"])
                for name in training._NAMES:
                    self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)))
                calls = len(self.calls)
                reused = self.run_fit(Model(), Path(root) / "resumed", seed)
                self.assertEqual(reused["records"], full["records"])
                self.assertEqual(calls, len(self.calls))

    def test_epoch_zero_wins_ties_and_patience_stops(self):
        self.tie = True
        with tempfile.TemporaryDirectory() as root:
            model = Model()
            result = self.run_fit(model, root)
            self.assertEqual((result["completed_epochs"], result["selected_epoch"]), (2, 0))
            self.assertTrue(torch.equal(model.correction_weight, torch.tensor([.2])))

    def test_changed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_fit(Model(), root, epoch_limit=0)
            self.config["learning_rate"] = .02
            with self.assertRaisesRegex(ValueError, "binding, protocol"):
                self.run_fit(Model(), root)

    def test_refit_has_no_validation_and_resumes_exact_budget(self):
        with tempfile.TemporaryDirectory() as root:
            args = (Model(), self.reference, self.loader, ["a"], [], self.config,
                    "refit", 43, root, {"input": "same"})
            first = training.run_epochs(*args, selected_epochs=2, epoch_limit=1)
            second = training.run_epochs(*args, selected_epochs=2)
            self.assertEqual((first["status"], second["status"]), ("PARTIAL", "COMPLETE"))
            self.assertEqual(second["completed_epochs"], 2)
            self.assertEqual(self.calls, ["a", "a"])

    def test_nonfinite_validation_keeps_previous_complete_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_fit(Model(), root, epoch_limit=0)
            checkpoint = Path(root) / "epoch_checkpoint.pt"
            initial = checkpoint.read_bytes()
            with patch.object(training, "residual_observation_mixture_loss",
                              return_value={"loss": torch.tensor(float("nan"))}):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.run_fit(Model(), root)
            self.assertEqual(checkpoint.read_bytes(), initial)

    def test_failed_partial_epoch_restarts_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            uninterrupted = self.run_fit(Model(), Path(root) / "full")
            interrupted = Path(root) / "interrupted"
            self.run_fit(Model(), interrupted, epoch_limit=0)
            calls = 0
            def failing(case):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated partial epoch interruption")
                return self.loader(case)
            with self.assertRaisesRegex(RuntimeError, "partial epoch"):
                training.run_epochs(Model(), self.reference, failing, ["a", "b"], ["v"],
                    self.config, "qualify", 41, interrupted, {"input": "same"})
            resumed = self.run_fit(Model(), interrupted)
            self.assertEqual(uninterrupted["records"], resumed["records"])

    def test_stream_domains_are_distinct_and_repeatable(self):
        values = [training._stream_seed(seed, phase, role, case)
                  for seed in (41, 42, 43) for phase in ("qualify", "refit")
                  for role in ("order", "train_initial", "train_dynamics",
                               "validation_initial", "validation_dynamics")
                  for case in ("", "v")]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(training._stream_seed(41, "qualify", "order"),
                         training._stream_seed(41, "qualify", "order"))


if __name__ == "__main__":
    unittest.main()

def test_native_epoch_optimizer_integration_all_seed_modes(tmp_path):
    from .test_population_residual_observation import arguments
    from .test_population_residual_memory import predictor
    from .population_residual_observation_training import run_epochs
    for seed in (41, 42, 43):
        for mode in ("current", "history"):
            batch = arguments(mode=mode, seed=seed, dtype=torch.float32)
            model = batch.pop("model")
            reference = predictor(mode, memory_gain=.15, constant=.01)
            frozen = {name: value.clone() for name, value in model.parent.state_dict().items()}
            config = dict(learning_rate=.0001, regularization_weight=.0001, max_epochs=2,
                          patience=2, prediction_particles=8, unit_chunk_size=2)
            report = run_epochs(model, reference, lambda case: batch, ["fit"], ["validation"],
                config, "qualify", seed, tmp_path / str(seed) / mode, {"mode": mode}, epoch_limit=1)
            assert report["status"] == "PARTIAL"
            assert report["completed_epochs"] == 1
            assert len(report["records"]) == 2
            assert all(torch.equal(value, model.parent.state_dict()[name]) for name, value in frozen.items())
            assert report["records"][1]["training"][0]["gradient_norm"] > 0


def test_new_fitter_imports_and_configuration(tmp_path):
    import yaml
    from .population_residual_observation_fitting import validate_config, cases
    config_path = Path(__file__).parents[1] / "configs/population_residual_observation_fitting.yaml"
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    assert len(cases(config["fitting_blocks"])) == 48
    assert len(cases(config["selection_blocks"])) == 8
    assert not set(cases(config["refit_blocks"])) & set(cases(config["excluded_blocks"]))

def test_native_refit_export_loader_and_dispatch(tmp_path):
    import json
    import yaml
    import pytest
    from .population_residual_observation_fitting import (
        SCHEMA, cases, json_digest, load_native_observation_correction, save_predictor)
    from .population_trajectory_evaluation import _apply_correction
    from .test_population_residual_memory import make_parent, predictor
    from .provenance import sha256_file
    parent = make_parent()
    binding = {"seed": 41}
    source_root = Path(__file__).parent
    config = yaml.safe_load((source_root.parent / "configs/population_residual_observation_fitting.yaml").read_text())
    reference = predictor("history")
    save_predictor(tmp_path / "predictor.npz", reference)
    identity = dict(seed=41, mode="history", config=config,
        parent_checkpoint_sha256="parent", parent_binding_sha256=json_digest(binding),
        fit_cases=cases(range(14)), validation_cases=[], selection_sha256="selection",
        sources={name: sha256_file(source_root / name) for name in
                 ("population_residual_memory.py", "population_state.py", "refractory_observation.py")})
    report = dict(schema=SCHEMA, status="FITTED_DEVELOPMENT", phase="refit", binding=identity,
        completed_epochs=1, selected_epoch=1, scientific_claim_qualified=False,
        predictor_sha256=sha256_file(tmp_path / "predictor.npz"))
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(report))
    loaded = _apply_correction(parent, binding, "parent", tmp_path)
    assert loaded.mode == "history"
    assert loaded.correction_identity["schema"] == SCHEMA
    report["phase"] = "qualify"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="refit"):
        load_native_observation_correction(tmp_path, parent, binding, "parent")
    report["phase"] = "refit"
    report["binding"]["seed"] = 42
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="refit"):
        load_native_observation_correction(tmp_path, parent, binding, "parent")

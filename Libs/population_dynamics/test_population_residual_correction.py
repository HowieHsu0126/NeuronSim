"""Artifact identity and fitting-selection tests for residual correction."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from population_dynamics import population_residual_correction as module
from population_dynamics.test_population_residual_memory import make_parent, predictor
from population_dynamics.provenance import sha256_file


def fitting():
    files = {"training/run.json": "b" * 64, "training/best.pt": "a" * 64}
    files.update({f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt": "c" * 64
                  for block in range(14) for template in range(4)})
    return dict(training_seeds=[41, 42, 43], fitting_blocks=list(range(14)),
                excluded_selection_blocks=list(range(14, 18)), diagnostic_sha256="d" * 64,
                coordinate_inputs=dict(files=files, selected_epoch=79))


class CorrectionArtifactTests(unittest.TestCase):
    def save(self, root, mode="history"):
        parent = make_parent()
        path = Path(root) / "correction"
        identity = module.save_correction(path, predictor(mode, memory_gain=.1), parent, seed=41,
            mode=mode, parent_checkpoint_sha256="a" * 64, fitting=fitting())
        return parent, path, identity

    def test_relocation_preserves_identity_and_affine_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            parent, path, identity = self.save(root)
            moved = Path(root) / "another_location"
            shutil.copytree(path, moved)
            model = module.load_correction(moved, parent, {"seed": 41}, "a" * 64)
            self.assertEqual(model.correction_identity, identity)
            z = torch.ones(2, 1, 2, 2)
            self.assertTrue(torch.allclose(model.correction(z, torch.zeros(1, 1), z), .1 * z))

    def test_rejects_parent_seed_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            parent, path, _ = self.save(root)
            for seed, digest in ((42, "a" * 64), (41, "f" * 64)):
                with self.subTest(seed=seed, digest=digest), self.assertRaisesRegex(ValueError, "identity differs"):
                    module.load_correction(path, parent, {"seed": seed}, digest)
            parent.field_names = ("changed", "B")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                module.load_correction(path, parent, {"seed": 41}, "a" * 64)

    def test_rejects_changed_arrays_and_implementation(self):
        with tempfile.TemporaryDirectory() as root:
            parent, path, _ = self.save(root)
            with patch.object(module, "_software", return_value={}):
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    module.load_correction(path, parent, {"seed": 41}, "a" * 64)
            with (path / "predictor.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "predictor changed"):
                module.load_correction(path, parent, {"seed": 41}, "a" * 64)

    def test_rejects_selection_cache_and_incomplete_fitting_inputs(self):
        for modification in ("selection", "missing"):
            with self.subTest(modification=modification), tempfile.TemporaryDirectory() as root:
                bad = fitting()
                files = bad["coordinate_inputs"]["files"]
                if modification == "selection":
                    files["training/posterior_cache/train_block_014_familiar_0.pt"] = "c" * 64
                else:
                    del files["training/posterior_cache/train_block_000_familiar_0.pt"]
                with self.assertRaisesRegex(ValueError, "cache identities"):
                    module.save_correction(Path(root) / "bad", predictor("current"), make_parent(),
                        seed=41, mode="current", parent_checkpoint_sha256="a" * 64, fitting=bad)

    def test_rejects_finite_but_wrong_predictor_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            parent, path, _ = self.save(root)
            arrays = {key: getattr(predictor("history"), key) for key in module.ARRAYS}
            arrays["coefficients"] = np.zeros((2, 2))
            np.savez(path / "predictor.npz", **arrays)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["identity"]["predictor_sha256"] = sha256_file(path / "predictor.npz")
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                module.load_correction(path, parent, {"seed": 41}, "a" * 64)


class RolloutSelectedCorrectionTests(unittest.TestCase):
    def qualification(self, root):
        """Small evidence fixtures test admission; they are not scientific runs."""
        qualification_module = module.rollout_qualification
        original = fitting()
        diagnostic = {"consumed": {str(seed): json.loads(json.dumps(original["coordinate_inputs"])) for seed in module.SEEDS}}
        runs = []
        for seed in module.SEEDS:
            for candidate, mode, alpha in qualification_module.CANDIDATES:
                complete = mode == "parent" or alpha in (1., 100.)
                score = 2. if mode == "parent" else ({1.: 1., 100.: 1.5}[alpha] if mode == "current" and complete
                         else ({1.: 1.2, 100.: .8}[alpha] if complete else None))
                origins = []
                for expected in qualification_module._origin_order():
                    origins.append(dict(expected, status="COMPLETE" if complete else "NOT_RUN",
                        completed_steps=100 if complete else 0, failure=None,
                        scores={str(h): dict(posterior_coordinate_energy_score=score) for h in (20, 80, 500)} if complete else {}))
                runs.append(dict(seed=seed, candidate=candidate, mode=mode, alpha=alpha,
                    status="COMPLETE" if complete else "FAILED", completed_origins=24 if complete else 0,
                    scored_origin_counts={str(h): 24 if complete else 0 for h in (20, 80, 500)},
                    mean_posterior_coordinate_energy_score={str(h): score for h in (20, 80, 500)}, origins=origins))
        consumed = {}
        for seed in module.SEEDS:
            coordinate_inputs = diagnostic["consumed"][str(seed)]
            consumed[str(seed)] = dict(authenticated_coordinates=coordinate_inputs,
                qualification_caches={key: value for key, value in coordinate_inputs["files"].items()
                                      if "train_block_012_" in key or "train_block_013_" in key})
        result = dict(module._qualification_protocol(), fitting_only_diagnostic_sha256=original["diagnostic_sha256"],
            software_sources=module._qualification_software(), score_rows=1944, consumed=consumed,
            coordinate_scales={str(seed): [1.] * 512 for seed in module.SEEDS}, runs=runs,
            **qualification_module.aggregate_candidates(runs))
        directory = Path(root) / "qualification"
        (directory / "candidates").mkdir(parents=True)
        artifacts = {}
        for run in runs:
            relative = f"candidates/seed_{run['seed']}_{run['candidate']}.json"
            artifact = directory / relative
            artifact.write_text(json.dumps(run))
            artifacts[relative] = sha256_file(artifact)
        (directory / "score_rows.csv").write_text("fixture_only\n")
        artifacts["score_rows.csv"] = sha256_file(directory / "score_rows.csv")
        result["artifacts"] = artifacts
        path = directory / "analysis.json"
        path.write_text(json.dumps(result))
        return path, diagnostic, original["diagnostic_sha256"], result

    def test_qualified_alphas_roundtrip_after_relocation(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            path, diagnostic, digest, _ = self.qualification(root)
            admitted = module.admit_rollout_qualification(path, diagnostic, digest)
            self.assertEqual(admitted["alpha_by_mode"], {"current": 1., "history": 100.})
            parent = make_parent()
            for mode in module.MODES:
                output = Path(root) / mode
                selected = admitted["regularization_selection_by_mode"][mode]
                identity = module.save_correction(output,
                    replace(predictor(mode, memory_gain=.1), alpha=admitted["alpha_by_mode"][mode]), parent,
                    seed=41, mode=mode, parent_checkpoint_sha256="a" * 64, fitting=fitting(),
                    regularization_selection=selected)
                moved = Path(root) / (mode + "_relocated")
                shutil.copytree(output, moved)
                restored = module.load_correction(moved, parent, {"seed": 41}, "a" * 64)
                self.assertEqual(restored.correction_identity, identity)
                self.assertTrue(identity["regularization_selection"]["refit_requires_native_evaluation"])
                self.assertEqual(identity["regularization_selection"]["protocol"]["correction_fitting_blocks"], list(range(12)))
                self.assertEqual(identity["fitting"]["fitting_blocks"], list(range(14)))

    def test_alternate_alpha_requires_matching_qualified_mode_and_source(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            parent = make_parent()
            with self.assertRaisesRegex(ValueError, "requires authenticated"):
                module.save_correction(Path(root) / "unqualified", replace(predictor("current"), alpha=1.), parent,
                    seed=41, mode="current", parent_checkpoint_sha256="a" * 64, fitting=fitting())
            path, diagnostic, digest, _ = self.qualification(root)
            admitted = module.admit_rollout_qualification(path, diagnostic, digest)
            selected = admitted["regularization_selection_by_mode"]["current"]
            for mutation in ("mode", "alpha", "source", "eligibility"):
                bad = json.loads(json.dumps(selected))
                if mutation == "mode":
                    bad["mode"] = "history"
                elif mutation == "alpha":
                    bad["selected"]["alpha"] = 100.
                elif mutation == "source":
                    bad["coordinate_input_sha256_by_seed"]["41"] = "f" * 64
                else:
                    bad["eligible_training_seeds"] = [41, 42]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    module.save_correction(Path(root) / mutation, replace(predictor("current"), alpha=1.), parent,
                        seed=41, mode="current", parent_checkpoint_sha256="a" * 64, fitting=fitting(), regularization_selection=bad)

    def test_rejects_absent_incomplete_changed_or_mismatched_qualification(self):
        for mutation in ("absent", "incomplete", "missing_run", "selection", "cache", "software", "artifact", "noise", "clipping"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                path, diagnostic, digest, result = self.qualification(root)
                if mutation == "absent":
                    path.unlink()
                elif mutation == "incomplete":
                    result["status"] = "RUNNING"
                elif mutation == "missing_run":
                    result["runs"].pop()
                elif mutation == "selection":
                    result["selected_alphas"]["current"]["alpha"] = 100.
                elif mutation == "cache":
                    result["consumed"]["42"]["qualification_caches"]["training/posterior_cache/train_block_012_familiar_0.pt"] = "f" * 64
                elif mutation == "software":
                    result["software_sources"]["population_state.py"] = "f" * 64
                elif mutation == "artifact":
                    (path.parent / "score_rows.csv").write_text("changed\n")
                elif mutation == "noise":
                    result["noise_stream_seeds"]["dynamics"] = 43
                else:
                    result["clipping_or_stabilization_applied"] = True
                if mutation != "absent":
                    path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    module.admit_rollout_qualification(path, diagnostic, digest)

    def test_no_three_seed_eligible_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, diagnostic, digest, result = self.qualification(root)
            for run in result["runs"]:
                if run["seed"] == 43 and run["mode"] == "current":
                    run["status"] = "FAILED"
                    run["completed_origins"] = 0
            result.update(module.rollout_qualification.aggregate_candidates(result["runs"]))
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "no eligible"):
                module.admit_rollout_qualification(path, diagnostic, digest)

    def test_loader_rechecks_selection_metadata(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            path, diagnostic, digest, _ = self.qualification(root)
            selected = module.admit_rollout_qualification(path, diagnostic, digest)["regularization_selection_by_mode"]["current"]
            output, parent = Path(root) / "correction", make_parent()
            module.save_correction(output, replace(predictor("current"), alpha=1.), parent, seed=41,
                mode="current", parent_checkpoint_sha256="a" * 64, fitting=fitting(), regularization_selection=selected)
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["identity"]["regularization_selection"]["selected"]["alpha"] = 100.
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not qualify"):
                module.load_correction(output, parent, {"seed": 41}, "a" * 64)

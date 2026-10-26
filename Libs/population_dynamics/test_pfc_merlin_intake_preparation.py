"""Intake preparation tests; no source simulation or network calls."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_PATH = Path(__file__).parents[1] / "scripts" / "prepare_pfc_merlin_scientific_intake.py"
_SPEC = importlib.util.spec_from_file_location("pfc_intake_preparation", _PATH)
preparation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preparation)


class IntakePreparationTests(unittest.TestCase):
    def write(self, path, value):
        path.write_bytes(preparation.json_bytes(value))

    def fixture(self, root):
        template = root / "template"
        template.mkdir()
        domain = root / "canonical_domain"
        source = "hdfs://checkpoints/original"
        task = dict(index=622, run_id="original", trial_id="trial-123", xie_job="643651_622",
                    source_root=str(domain / "source_results"), receipt_uri=source + "/pfc/task_0622/receipt.json",
                    bundle_uri=source + "/pfc/task_0622/result.tar.gz", output_bundle="retained.tar.gz",
                    ready_evidence="queue.json")
        config = dict(domain_base=str(domain), prior_acceptance=str(root / "acceptance_637.json"),
                      return_tasks=[task], control_validation="frozen-control.json",
                      numerical_source="frozen-numerical")
        self.write(template / "config.json", config)
        base = ('#!/bin/bash\nset -euo pipefail\n'
                'mkdir "$NRNSIM_MERLIN_INTAKE_WORK"\n'
                'echo base_config > "$NRNSIM_MERLIN_INTAKE_WORK/config.json"\n'
                + preparation.FINAL_EXEC + '\n').encode()
        (template / "entrypoint.sh").write_bytes(base)
        self.write(template / "entrypoint_upload.json", dict(hdfs_path="hdfs://entrypoint/base.sh",
                   bytes=len(base), sha256=preparation.sha256(base)))
        request = {
            "name": "mrln_original",
            "job_config": {"job_template_config": {
                "entrypoint_full_script": "",
                "image_meta": {"image_url": "pinned:image", "image_source": "url"},
                "env_map": {"PYTHONDONTWRITEBYTECODE": "1"}}},
            "resource_config": {"arnold_resource_config": {"roles": [
                {"name": "head", "num": 1, "cpu": 2, "gpu": 0,
                 "memory_mb": 4096, "gpu_type": "CPU_ONLY"}]}}
        }
        self.write(template / "request_template.json", request)
        prior = root / "acceptance_654.json"
        self.write(prior, dict(schema="pfc_merlin_scientific_intake_aggregation_v1", status="PASS",
                   accepted_task_indices=[0, 622], accepted_episode_count=2, production_episodes=0))
        ready = root / "ready.json"
        self.write(ready, dict(run_id="original", trial_id="trial-123",
                   original_progress_uri=source + "/pfc/progress.json", completed=[
                       dict(index=index, run_id="original", trial_id="trial-123", exit_code=0,
                            persisted=True, checkpoint_path=source + f"/pfc/task_{index:04d}")
                       for index in (623, 1005, 2005)]))
        return template, prior, ready, root / "output"

    def test_selection_offsets_serialization_and_effective_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, prior, ready, output = self.fixture(root)
            old = (template / "entrypoint.sh").read_bytes()
            result = preparation.prepare(template, prior, ready, "623,1005,2005", "mrln_intake", output)
            self.assertEqual(result["selected_episode_count"], 3)
            request = json.loads((output / "request.json").read_bytes())
            self.assertLess(len((output / "request.json").read_bytes()), 1024 * 1024)
            check = json.loads((output / "check.json").read_bytes())
            self.assertEqual(check, dict(check_point="create", node_type="merlin_job_run",
                                         job_run_context=request))
            self.assertEqual(json.loads((output / "execution.json").read_bytes())["status"],
                             "PREPARED_NOT_SUBMITTED")
            self.assertEqual(request["job_config"]["job_template_config"]["image_meta"]["image_url"], "pinned:image")
            config = json.loads((output / "config.json").read_bytes())
            self.assertEqual([task["xie_job"] for task in config["return_tasks"]],
                             ["643651_623", "643652_5", "643653_5"])
            self.assertEqual(config["prior_acceptance"], str(prior))
            self.assertEqual((template / "entrypoint.sh").read_bytes(), old)
            runtime = (output / "runtime_preparation.py").read_text()
            ast.parse(runtime)
            work, bootstrap = root / "work", root / "bootstrap"
            bootstrap.mkdir()
            (bootstrap / "base_entrypoint.sh").write_bytes(old)
            with patch.dict(os.environ, NRNSIM_MERLIN_INTAKE_WORK=str(work),
                            NRNSIM_MERLIN_INTAKE_BOOTSTRAP=str(bootstrap)):
                exec(compile(runtime, "generated_runtime", "exec"), {})
            self.assertEqual((work / "config.json").read_bytes(), (output / "config.json").read_bytes())
            self.assertEqual(prior.read_bytes(), (output / "prior_acceptance.json").read_bytes())
            self.assertNotIn(preparation.FINAL_EXEC, (bootstrap / "base_preparation.sh").read_text())
            self.assertTrue(request["job_config"]["job_template_config"]["entrypoint_full_script"].endswith(
                            preparation.FINAL_EXEC + "\n"))

    def test_invalid_inputs_reject_before_output_creation(self):
        cases = ("duplicate", "accepted", "range", "not_ready", "prior_count", "prior_duplicate",
                 "prior_failed", "production", "name", "ready_failed", "ready_not_persisted",
                 "ready_duplicate", "run", "trial", "checkpoint", "template_changed")
        for kind in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                template, prior, ready, output = self.fixture(root)
                indices, name = "623", "mrln_intake"
                if kind == "duplicate": indices = "623,623"
                elif kind == "accepted": indices = "622"
                elif kind == "range": indices = "3000"
                elif kind == "not_ready": indices = "624"
                elif kind == "name": name = "pfc_intake"
                elif kind.startswith("prior") or kind == "production":
                    value = json.loads(prior.read_bytes())
                    if kind == "prior_count": value["accepted_episode_count"] = 3
                    elif kind == "prior_duplicate": value["accepted_task_indices"] = [0, 0]
                    elif kind == "prior_failed": value["status"] = "FAILED"
                    else: value["production_episodes"] = 1
                    self.write(prior, value)
                elif kind == "template_changed":
                    with (template / "entrypoint.sh").open("ab") as stream:
                        stream.write(b"changed")
                else:
                    value = json.loads(ready.read_bytes())
                    if kind == "ready_failed": value["completed"][0]["exit_code"] = 1
                    elif kind == "ready_not_persisted": value["completed"][0]["persisted"] = False
                    elif kind == "ready_duplicate": value["completed"].append(dict(value["completed"][0]))
                    elif kind == "run": value["completed"][0]["run_id"] = "other"
                    elif kind == "trial": value["completed"][0]["trial_id"] = "other"
                    elif kind == "checkpoint": value["completed"][0]["checkpoint_path"] = "hdfs://other"
                    self.write(ready, value)
                with self.assertRaises(ValueError):
                    preparation.prepare(template, prior, ready, indices, name, output)
                self.assertFalse(output.exists())

    def test_cli_invalid_identity_rejects_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, prior, ready, output = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "MRLN_JOB_NAME_REQUIRED"):
                preparation.main(["--template", str(template), "--prior-acceptance", str(prior),
                                  "--completion-index", str(ready), "--indices", "623",
                                  "--name", "wrong", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_runtime_rejects_changed_uploaded_entrypoint_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, prior, ready, output = self.fixture(root)
            preparation.prepare(template, prior, ready, "623", "mrln_intake", output)
            bootstrap = root / "bootstrap"
            bootstrap.mkdir()
            (bootstrap / "base_entrypoint.sh").write_bytes(b"changed")
            with patch.dict(os.environ, NRNSIM_MERLIN_INTAKE_WORK=str(root / "work"),
                            NRNSIM_MERLIN_INTAKE_BOOTSTRAP=str(bootstrap)), \
                    patch("subprocess.run") as run, \
                    self.assertRaisesRegex(ValueError, "BASE_ENTRYPOINT_CHANGED"):
                exec(compile((output / "runtime_preparation.py").read_text(), "generated_runtime", "exec"), {})
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

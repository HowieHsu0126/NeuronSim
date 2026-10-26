"""Portable refit inventory checks; no model or remote execution."""
import ast
import base64
from contextlib import redirect_stdout
import gzip
import importlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import prepare_population_residual_generation_merlin as generation
from prepare_population_residual_generation_merlin import fingerprint, refit_inventory


class RefitInventoryTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for seed in (41, 42, 43):
            for mode in ("current", "history"):
                for name in ("fit.json", "predictor.npz"):
                    relative = f"seed_{seed}/{mode}/{name}"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(relative.encode())
                    rows.append(dict(path=relative, **fingerprint(path)))
        return rows

    def test_exact_export_inventory_is_accepted_and_missing_cell_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            refit_inventory(root, rows)
            with self.assertRaisesRegex(ValueError, "exactly six"):
                refit_inventory(root, rows[:-1])

    def test_inventory_cannot_replace_a_canonical_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            rows[-1]["sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "canonical export"):
                refit_inventory(root, rows)


class GenerationSelectionTests(unittest.TestCase):
    """Exercise CLI request preparation with six synthetic fit receipts."""

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def fixture(self, root, *, mean=False):
        campaign = root / "campaign"
        fitting = campaign / "native_observation_fitting"
        checks = root / "mean_source_checks" if mean else fitting / "checkpoint_checks/native_route_recovery"
        refits = root / "mean_refits" if mean else fitting / "refits"
        config_relative = generation.MEAN_CONFIG if mean else generation.DEFAULT_CONFIG
        config = dict(training_seeds=[41, 42, 43], max_epochs=80, cpu_threads=2,
                      fixture_objective="mean" if mean else "original")
        repo = root / "repo"
        for name in ("__init__.py", "scripts/run_population_residual_correction.py",
                     "scripts/run_population_trajectory_campaign.py", "population_dynamics/population_residual_refit_provenance.py"):
            target = repo / "Libs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('"""Synthetic prepared source; never executed by this fixture."""\n')
        config_path = repo / "Libs" / config_relative
        self.write_json(config_path, config)
        sources = {config_relative: fingerprint(config_path)}
        names = ["population_residual_observation_fitting.py"]
        if mean:
            names += ["population_residual_mean_objective.py", "native_renewal_expectation.py",
                      "population_residual_observation_training.py", "population_residual_observation_execution.py",
                      "poisson_trajectory_evaluation.py"]
        bindings = {}
        for name in names:
            target = repo / "Libs/population_dynamics" / name
            target.write_text('"""Synthetic fitting source identity: ' + name + '."""\n')
            sources["population_dynamics/" + name] = fingerprint(target)
            bindings[name] = sources["population_dynamics/" + name]["sha256"]
        source = dict(complete=True, bytes=101, sha256="a" * 64, hdfs_path="hdfs://fixture/source.tar.gz")
        self.write_json(checks / "source_upload.json", source)
        self.write_json(checks / "result.json", dict(status="PASS", exit_code=0, tests=35,
            failures=0, errors=0, skipped=0, sources=sources, source_payload_sha256=source["sha256"]))
        self.write_json(checks / "request.json", dict(request_context={"discard": True},
            job_config=dict(job_template_config=dict(entrypoint_full_script="CPU check fixture",
                env_map=dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"))),
            resource_config=dict(arnold_resource_config=dict(cluster_ids=[9], group_id=9,
                roles=[dict(cpu=2, memory_mb=4096, gpu_type="CPU_ONLY", queue_name="source_checks")]))))
        for name, admission_relative, manifest_relative, count in (
            ("fitting", "input_admission/complete_inputs/analysis.json", "real_inputs/full_manifest.json", 462),
            ("generation", "generation_inputs/complete_admission/analysis.json", "generation_inputs/additional_manifest.json", 81)):
            self.write_json(fitting / manifest_relative, dict(fixture=name))
            self.write_json(fitting / admission_relative, dict(status="VERIFIED_INPUT_BUNDLE", exit_code=0,
                input_manifest_sha256=fingerprint(fitting / manifest_relative)["sha256"], logical_files=count,
                archive=dict(bytes=100, sha256="b" * 64, hdfs_path="hdfs://fixture/" + name + ".tar.gz")))
        self.write_json(campaign / "merlin_readout_analysis/wrapper_upload.json",
                        dict(bytes=10, sha256="c" * 64, hdfs_path="hdfs://fixture/publisher.py"))
        input_digest = fingerprint(fitting / "real_inputs/full_manifest.json")["sha256"]
        inventory = []
        for seed in (41, 42, 43):
            for mode in ("current", "history"):
                cell = refits / f"seed_{seed}" / mode
                cell.mkdir(parents=True)
                (cell / "predictor.npz").write_bytes(f"synthetic predictor {seed} {mode}".encode())
                report = dict(schema="native_residual_observation_fit_v1", status="FITTED_DEVELOPMENT", phase="refit",
                    completed_epochs=1, selected_epoch=1, parent_unchanged=True, scientific_claim_qualified=False,
                    predictor_sha256=fingerprint(cell / "predictor.npz")["sha256"], run_id="fixture_run", trial_id="fixture_trial",
                    binding=dict(seed=seed, mode=mode, phase="refit", config=config, validation_cases=[],
                        selection_sha256="d" * 64, parent_checkpoint_sha256=f"{seed:064x}",
                        input_manifest_sha256=input_digest, config_sha256=sources[config_relative]["sha256"], sources=bindings))
                self.write_json(cell / "fit.json", report)
                for name in ("fit.json", "predictor.npz"):
                    inventory.append(dict(path=f"seed_{seed}/{mode}/{name}", **fingerprint(cell / name)))
        inventory_path, upload_path = root / "refits_inventory.json", root / "refits_upload.json"
        self.write_json(inventory_path, inventory)
        self.write_json(upload_path, dict(complete=True, bytes=100, sha256="e" * 64, chunks=[]))
        output = root / "prepared"
        argv = ["--campaign-root", str(campaign), "--repo", str(repo), "--output", str(output),
                "--refits-upload", str(upload_path), "--refits-inventory", str(inventory_path),
                "--name", "mrln_generation_fixture", "--seed", "42", "--cluster-id", "1", "--group-id", "2",
                "--queue", "gpu_fixture", "--gpu-type", "A100_SXM_80GB"]
        if mean:
            argv += ["--source-checks", str(checks), "--config-relative", config_relative, "--refit-root", str(refits)]
        return dict(argv=argv, output=output, campaign=campaign, fitting=fitting, checks=checks,
                    refits=refits, config_relative=config_relative, source=source)

    def invoke(self, fixture):
        # Import the real family validator. The fixture repo only supplies bytes
        # for request overlays; none of its synthetic source is executed.
        library = str(Path(__file__).resolve().parents[1])
        with patch.object(sys, "path", [library, *sys.path]), redirect_stdout(io.StringIO()):
            importlib.import_module("population_dynamics.population_residual_refit_provenance")
            generation.main(fixture["argv"])
        request = json.loads((fixture["output"] / "request.json").read_text())
        script = request["job_config"]["job_template_config"]["entrypoint_full_script"]
        python = script.split("python3 - <<'NATIVE_RESIDUAL_GENERATION'\n", 1)[1].rsplit("\nNATIVE_RESIDUAL_GENERATION", 1)[0]
        tree = ast.parse(python)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "payload")
        encoded = assignment.value.args[0].args[0].args[0].value
        payload = json.loads(gzip.decompress(base64.b64decode(encoded)))
        return request, payload, tree

    def test_default_cli_preserves_original_inputs_publisher_and_two_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            request, payload, _ = self.invoke(fixture)
            execution = json.loads((fixture["output"] / "execution.json").read_text())
            self.assertEqual(payload["config_relative"], generation.DEFAULT_CONFIG)
            self.assertEqual(execution["refit_root"], str(fixture["refits"].resolve()))
            self.assertEqual(execution["source_checks"], str(fixture["checks"].resolve()))
            self.assertEqual(payload["source"], fixture["source"])
            self.assertEqual(payload["publisher"]["hdfs_path"], "hdfs://fixture/publisher.py")
            for name in ("fitting", "generation"):
                self.assertEqual(payload["bundles"][name]["chunks"][0]["hdfs_path"], "hdfs://fixture/" + name + ".tar.gz")
            env = request["job_config"]["job_template_config"]["env_map"]
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertEqual({key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                             dict(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
            self.assertEqual(execution["cpu_threads"], 2)
            self.assertEqual(execution["source_units"], 100000)
            self.assertEqual(execution["fields"], 128)
            self.assertFalse(execution["submitted"])
            self.assertEqual(len(payload["refit_receipt"]["fits"]), 6)

    def test_mean_cli_uses_explicit_refits_and_selected_config_in_child_command(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), mean=True)
            self.assertFalse((fixture["fitting"] / "refits").exists())
            self.assertFalse((fixture["fitting"] / "checkpoint_checks/native_route_recovery").exists())
            request, payload, tree = self.invoke(fixture)
            self.assertEqual(payload["config_relative"], generation.MEAN_CONFIG)
            self.assertEqual(payload["config_identity"]["sha256"], payload["refit_receipt"]["shared_binding"]["config_sha256"])
            # Evaluate only the generated command-list expression, never the
            # bootstrap, publisher, model, or subprocess.
            assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "command")
            command = eval(compile(ast.Expression(assignment.value), "<prepared-command>", "eval"),
                           {"sys": sys, "repo": Path("/staged/repo"), "root": Path("/staged"),
                            "output": Path("/staged/delivery"), "payload": payload})
            self.assertEqual(command[command.index("--fitting-config") + 1], "/staged/repo/Libs/" + generation.MEAN_CONFIG)
            self.assertEqual(command[command.index("--native-refits") + 1], "/staged/refits/inputs")
            self.assertEqual(command[command.index("--seed") + 1], "42")
            resource = request["resource_config"]["arnold_resource_config"]
            self.assertEqual((resource["cluster_ids"], resource["group_id"]), ([1], 2))
            self.assertEqual(resource["roles"][0]["gpu"], 1)

    def test_config_or_source_mismatch_rejects_before_preparing_output(self):
        for mismatch in ("config", "source"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(Path(directory), mean=True)
                path = fixture["checks"] / "result.json"
                result = json.loads(path.read_text())
                relative = fixture["config_relative"] if mismatch == "config" else "population_dynamics/population_residual_observation_fitting.py"
                result["sources"][relative]["sha256"] = "f" * 64
                self.write_json(path, result)
                with self.assertRaisesRegex(ValueError, "native refits differ"):
                    self.invoke(fixture)
                self.assertFalse(fixture["output"].exists())

    def test_incomplete_sixth_refit_rejects_before_preparing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), mean=True)
            path = fixture["refits"] / "seed_43/history/fit.json"
            report = json.loads(path.read_text())
            report["status"] = "RUNNING"
            self.write_json(path, report)
            with self.assertRaisesRegex(ValueError, "complete Merlin native observation refit metadata"):
                self.invoke(fixture)
            self.assertFalse(fixture["output"].exists())


if __name__ == "__main__":
    unittest.main()

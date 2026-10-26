"""Metadata admission boundaries for the existing comparison CLI."""
from copy import deepcopy
import unittest

from prepare_population_residual_comparison_merlin import admit_metadata, extract_field_arrays, field_array_paths


class AdmissionTests(unittest.TestCase):
    def fixtures(self):
        receipt = dict(persisted=True, exit_code=0)
        rows = [dict(path="delivery/delivery.json", bytes=1, sha256="delivery")]
        export = dict(status="VERIFIED_GENERATION_METADATA", exit_code=0, seed=41, source_run="run", source_trial="trial",
            scope="comparison_metadata_return", large_arrays_returned=False, scientific_computation_repeated=False,
            full_output_persisted=True, delivery_sha256="delivery", returned_files=rows, complete_modes=["current", "history"])
        intake = dict(status="VERIFIED_GENERATION_METADATA", seed=41, files=1, large_arrays_returned=False,
            source_run="run", delivery_sha256="delivery", modes={mode: dict(status="PASS", exit_code=0) for mode in ("current", "history")})
        return receipt, export, intake

    def test_accepts_metadata_and_rejects_wrong_seed_or_source_run(self):
        values = self.fixtures()
        self.assertEqual(admit_metadata(41, *values)["seed"], 41)
        with self.assertRaises(ValueError):
            admit_metadata(42, *values)
        changed = deepcopy(values)
        changed[2]["source_run"] = "another-run"
        with self.assertRaises(ValueError):
            admit_metadata(41, *changed)

    def test_rejects_failed_mode_and_nonmetadata_file(self):
        values = self.fixtures()
        values[2]["modes"]["history"]["status"] = "FAILED"
        with self.assertRaises(ValueError):
            admit_metadata(41, *values)
        values = self.fixtures()
        values[1]["returned_files"][0]["path"] = "delivery/array.npz"
        with self.assertRaises(ValueError):
            admit_metadata(41, *values)


class FieldArchiveTests(unittest.TestCase):
    def test_only_complete_manifest_bound_field_arrays_are_extracted(self):
        import hashlib
        import io
        from pathlib import Path
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"field-fixture"
            rows = [dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                    for name in sorted(field_array_paths())]
            archive = root / "generation.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name in [row["path"] for row in rows] + ["results/current/forecasts/full_unit_samples.npz"]:
                    member = tarfile.TarInfo("results/delivery/" + name)
                    member.size = len(payload)
                    bundle.addfile(member, io.BytesIO(payload))
            def verify(path, row):
                self.assertEqual(path.stat().st_size, row["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            result = extract_field_arrays(archive, root / "received", rows, verify)
            self.assertEqual(len(result), 98)
            self.assertFalse((root / "received/results/current/forecasts/full_unit_samples.npz").exists())
            with self.assertRaisesRegex(ValueError, "complete unique"):
                extract_field_arrays(archive, root / "partial", rows[:-1], verify)

    def test_missing_or_unsafe_generation_member_rejected(self):
        import io
        from pathlib import Path
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(path=name, bytes=0, sha256="unused") for name in field_array_paths()]
            for unsafe in (False, True):
                archive = root / (str(unsafe) + ".tar.gz")
                with tarfile.open(archive, "w:gz") as bundle:
                    if unsafe:
                        member = tarfile.TarInfo("../outside")
                        member.size = 0
                        bundle.addfile(member, io.BytesIO(b""))
                with self.assertRaisesRegex(ValueError, "unsafe|lacks required"):
                    extract_field_arrays(archive, root / str(unsafe), rows, lambda *args: None)


class LdnsPreparationTests(unittest.TestCase):
    def _prepare_route(self, decomposition=False, *, selected=False, rejection=None, eag=False):
        import ast
        import base64
        import contextlib
        import gzip
        import io
        import json
        from pathlib import Path
        import tempfile
        from prepare_population_residual_comparison_merlin import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, repo, output = (root / name for name in ("campaign", "repo", "prepared"))
            fitting = campaign / "native_observation_fitting"
            inputs = fitting / "comparison_inputs" if eag else root / "inputs"
            generation = root / "mean_generation" if selected else fitting / "generation"
            checks = root / "mean_source_checks" if selected else fitting / "checkpoint_checks/native_route_recovery"
            request_template = root / "mean_request.json" if selected else fitting / "generation_checks/request.json"

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            for seed in (41, 42, 43):
                receipt, export, intake = AdmissionTests().fixtures()
                export["seed"] = intake["seed"] = seed
                if selected:
                    export["source_run"] = intake["source_run"] = f"mean_generation_{seed}"
                if decomposition:
                    export["source_archive"] = dict(persisted=True, exit_code=0, result_bytes=1, result_sha256="original",
                                                    checkpoint_path="hdfs://fixture/original")
                if rejection == "metadata" and seed == 43:
                    intake["modes"]["history"]["status"] = "FAILED"
                cell = generation / f"seed_{seed}" / "metadata_return"
                for name, value in (("receipt.json", receipt), ("analysis.json", export), ("intake.json", intake)):
                    write(cell / name, value)
            descriptor = dict(complete=True, bytes=1, sha256="fixture", hdfs_path="hdfs://fixture")
            source_descriptor = (dict(descriptor, sha256="mean_source", hdfs_path="hdfs://fixture/mean_source")
                                 if selected else descriptor)
            write(checks / "source_upload.json", source_descriptor)
            if selected:
                validation = dict(status="PASS", exit_code=0, tests=6, failures=0, errors=0, skipped=0,
                                  source_payload_sha256=source_descriptor["sha256"])
                if rejection == "source":
                    validation["source_payload_sha256"] = "different_source"
                if rejection == "validation":
                    validation["failures"] = 1
                write(checks / "result.json", validation)
            write(campaign / "merlin_readout_analysis/wrapper_upload.json", descriptor)
            template_label = "explicit_mean_fixture" if selected else "historical_fixture"
            write(request_template, dict(fixture_template=template_label,
                job_config=dict(job_template_config=dict(env_map={})),
                resource_config=dict(arnold_resource_config=dict(roles=[{}]))))
            names = {f"parent/seed_{seed}/{name}" for seed in (41, 42, 43)
                     for name in ("delivery.json", "forecasts/evaluation.json", "forecasts/scores.csv", "replay.json")}
            if eag:
                names |= {"eag/acceptance.json", "eag/comparison.json", "eag/joint_comparison_all_seeds.csv"}
            else:
                names |= {"ldns/" + name for name in ("delivery.json", "source_delivery.json", "source_command.json",
                           "analysis/comparison.json", "analysis/joint_comparison_all_seeds.csv")}
                names |= {"field/intake.json", "field/comparison.json"}
            if decomposition:
                import itertools
                names.add("field/fitting_field_scale.npz")
                names |= {f"ldns_forecasts/seed_{seed}/delivery.json" for seed in (41, 42, 43)}
                names |= {f"ldns_forecasts/seed_{seed}/forecasts/train_block_{block:03d}_familiar_{template}_origin_{origin}.npz"
                          for seed, block, template, origin in itertools.product((41, 42, 43), range(14, 18), range(4), (1000, 3000, 5000))}
                self.assertEqual(len(names), 167)
            inventory = [dict(path=name) for name in sorted(names)]
            if rejection == "inventory":
                inventory.pop()
            write(inputs / "input_inventory.json", inventory)
            write(inputs / "upload.json", descriptor)
            source_names = ["population_residual_correction_analysis.py", "population_residual_refit_provenance.py", "provenance.py"]
            if decomposition:
                source_names.extend(["population_native_forecast_decomposition.py", "trajectory_scores.py", "lif_neural_field.py",
                                     "braid_population_comparison.py"])
            for name in source_names:
                path = repo / "Libs/population_dynamics" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n")
            argv = ["--campaign-root", str(campaign), "--repo", str(repo),
                    "--output", str(output), "--name", "mrln_fixture_ldns"]
            if not eag:
                argv += ["--decomposition-inputs" if decomposition else "--ldns-inputs", str(inputs)]
            if selected:
                argv += ["--generation-root", str(generation), "--source-checks", str(checks),
                         "--request-template", str(request_template)]
                self.assertFalse((fitting / "generation").exists())
                self.assertFalse((fitting / "checkpoint_checks/native_route_recovery/source_upload.json").exists())
                self.assertFalse((fitting / "generation_checks/request.json").exists())
            if rejection is not None:
                message = {"source": "exact tested source payload", "validation": "exact tested source payload",
                           "metadata": "complete accepted metadata", "inventory": "complete declared comparison input bundle"}[rejection]
                with self.assertRaisesRegex(ValueError, message), contextlib.redirect_stdout(io.StringIO()):
                    main(argv)
                self.assertFalse(output.exists())
                return
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            execution = json.loads((output / "execution.json").read_text())
            request = json.loads((output / "request.json").read_text())
            expected_scope = ("comparison_metadata_only" if eag else
                              "native_forecast_decomposition" if decomposition else "native_ldns_comparison")
            self.assertEqual(execution["scope"], expected_scope)
            self.assertFalse(execution["submitted"])
            self.assertEqual(execution["generation_root"], str(generation.resolve()))
            self.assertEqual(execution["source_checks"], str(checks.resolve()))
            self.assertEqual(execution["request_template"], str(request_template.resolve()))
            self.assertEqual(request["fixture_template"], template_label)
            entry = request["job_config"]["job_template_config"]["entrypoint_full_script"]
            code = entry.split("python3 - <<'NATIVE_COMPARISON'\n", 1)[1].rsplit("\nNATIVE_COMPARISON", 1)[0]
            tree = ast.parse(code)
            encoded = next(node.value for node in ast.walk(tree)
                           if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("H4sI"))
            payload = json.loads(gzip.decompress(base64.b64decode(encoded)))
            self.assertEqual(payload["scope"], expected_scope)
            self.assertEqual(len(payload["metadata"]), 3)
            self.assertTrue(all(("source_archive" in item) == decomposition for item in payload["metadata"]))
            self.assertEqual(payload["source"], source_descriptor)
            self.assertEqual(payload["publisher"], descriptor)
            self.assertEqual(payload["inputs"], descriptor)
            if selected:
                self.assertEqual([item["export"]["source_run"] for item in payload["metadata"]],
                                 [f"mean_generation_{seed}" for seed in (41, 42, 43)])
            if decomposition:
                self.assertIn("--ldns-forecasts", code)
                self.assertIn("--field-scale", code)
                self.assertEqual(set(payload["overlays"]), set(source_names))
            self.assertIn("--ldns-comparison", code)
            self.assertIn("--field-comparison", code)
            self.assertEqual(request["resource_config"]["arnold_resource_config"]["roles"][0]["gpu_type"], "CPU_ONLY")

    def test_ldns_cli_prepares_metadata_route_without_generation_arrays(self):
        self._prepare_route()

    def test_decomposition_cli_binds_full_archives_and_all_saved_inputs(self):
        self._prepare_route(decomposition=True)


    def test_explicit_mean_paths_keep_baseline_inputs_and_publisher(self):
        for eag, decomposition in ((True, False), (False, False), (False, True)):
            with self.subTest(eag=eag, decomposition=decomposition):
                self._prepare_route(decomposition=decomposition, selected=True, eag=eag)

    def test_explicit_source_checks_reject_failed_or_unbound_source(self):
        for rejection in ("source", "validation"):
            with self.subTest(rejection=rejection):
                self._prepare_route(selected=True, rejection=rejection)

    def test_explicit_paths_preserve_metadata_and_inventory_admission(self):
        for rejection in ("metadata", "inventory"):
            with self.subTest(rejection=rejection):
                self._prepare_route(selected=True, rejection=rejection)



if __name__ == "__main__":
    unittest.main()

"""CUGA Merlin integration contracts; execute on Merlin CPU."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

from population_dynamics import test_appworld_benchmark_merlin as fixture
from population_dynamics import appworld_benchmark_cuga_process as cuga_process
from scripts import qualify_appworld_cuga_merlin_binding as binding_qualification
from population_dynamics.appworld_benchmark_cuga_process import _check_mounts
from population_dynamics.appworld_benchmark_cuga_receipt import CUGA_METHOD

bootstrap = fixture.bootstrap


def content_record(content: bytes) -> dict:
    return {
        "content_base64": base64.b64encode(content).decode(),
        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
    }


def cuga_attempt(**changes) -> dict:
    return {"agent": "cuga", "split": "train", "eligible_index": 0, "seed": 41, **changes}


def cuga_inputs() -> dict:
    return {
        **fixture.valid_inputs(),
        "development_attempt": cuga_attempt(),
        "cuga_runtime": {
            "schema": "appworld_cuga_merlin_preparation_v1",
            "persisted": True, "runtime_prepared": True, "runtime_persisted": True,
            "exit_code": 1,
        },
        "cuga_prior_launcher": {
            "bytes": bootstrap.CUGA_PRIOR_LAUNCHER_BYTES,
            "sha256": bootstrap.CUGA_PRIOR_LAUNCHER_SHA256,
        },
        "files": {name: content_record(b"fixture source") for name in bootstrap.CUGA_SOURCE_FILES},
    }


class CugaDeclarationTests(fixture.TemporaryArtifacts):
    def test_persisted_runtime_can_outlive_failed_structural_checks(self):
        inputs = cuga_inputs()
        bootstrap.validate_inputs(inputs)
        self.assertEqual(bootstrap.resolve_development_attempt(inputs)["method"], CUGA_METHOD)
        self.assertEqual(bootstrap.CUGA_METHOD, CUGA_METHOD)
        bootstrap.validate_inputs(fixture.valid_inputs())

    def test_invalid_cuga_declarations_and_restore_inputs_are_rejected(self):
        for changes in ({"seed": 44}, {"seed": True}, {"split": "test_normal"},
                        {"max_model_calls": 50}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bootstrap.resolve_development_attempt({"development_attempt": cuga_attempt(**changes)})
        for key in ("cuga_runtime", "cuga_prior_launcher"):
            inputs = cuga_inputs()
            del inputs[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                bootstrap.validate_inputs(inputs)
        inputs = cuga_inputs()
        inputs["cuga_runtime"]["runtime_persisted"] = False
        with self.assertRaisesRegex(ValueError, "PERSISTED_CUGA_RUNTIME"):
            bootstrap.validate_inputs(inputs)
        inputs = cuga_inputs()
        inputs["cuga_prior_launcher"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PINNED_CUGA"):
            bootstrap.validate_inputs(inputs)
        inputs = cuga_inputs()
        del inputs["files"]["Libs/population_dynamics/appworld_benchmark_cuga_receipt.py"]
        with self.assertRaisesRegex(ValueError, "CUGA_DEVELOPMENT_SOURCE"):
            bootstrap.validate_inputs(inputs)

    def test_cuga_results_and_public_code_do_not_overlap_private_repository(self):
        private = self.work / "private"
        normal = bootstrap.development_output_root(private, [fixture.official_attempt()])
        output = bootstrap.development_output_root(private, [cuga_attempt()])
        public = private.with_name(private.name + "-public") / "Libs"
        self.assertEqual(normal, private / "artifacts")
        self.assertEqual(output.parent, private.parent)
        self.assertNotEqual(output, private)
        protected = [private, output / "native_run", output / "development_score.json"]
        _check_mounts([public], protected)
        _check_mounts([output / "cuga_process/sandbox"], protected)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            _check_mounts([normal / "cuga_process/sandbox"], protected)

    def test_batch_registers_generic_and_cuga_owners_separately(self):
        with mock.patch.object(
            bootstrap, "prepare_development_context", side_effect=ValueError("fixture stop"),
        ):
            with self.assertRaisesRegex(ValueError, "fixture stop"):
                bootstrap.run_development_batch(
                    sys.executable, sys.executable, {}, self.work, self.output,
                    self.store, self.report,
                    development_attempts=[fixture.official_attempt(), cuga_attempt()],
                )
        self.assertEqual(self.store.native_attempt_roots, (
            "development/official_full_code/train/seed_41/eligible_0/attempt",
        ))
        self.assertEqual(self.store.native_session_roots, (
            "development/cuga/train/seed_41/eligible_0/native_run",
        ))



def successful_cuga_binding(binary: dict) -> dict:
    return {
        "schema": "appworld_cuga_merlin_binding_check_v1", "status": "PASS",
        "exit_code": 0, "persisted": True, "execution_backend": "merlin",
        "allocated_gpus": 0, "run_id": "fixture-cpu-run", "trial_id": "fixture-cpu-trial",
        "hostname": "fixture-cpu-host", "native_tasks_executed": 0, "model_generations": 0,
        "graph_loop_executed": False, "formal_eligible": False,
        "bubblewrap": copy.deepcopy(binary), "public_import": {"status": "PASS"},
        "sandbox_probe": {
            "command": [binary["executable"], "--unshare-all", "--share-net", "--cap-drop", "ALL"],
            "exit_code": 0, "sandbox_executed": True, "bubblewrap_sha256": binary["sha256"],
            "isolation_checks": {key: True for key in (
                "private_repository_hidden", "procfs_absent", "work_directory", "public_modules_only",
            )},
        },
    }


class CugaBubblewrapTests(fixture.TemporaryArtifacts):
    def setUp(self):
        super().setUp()
        self.binary = self.work / "fixture-bwrap"
        self.binary.write_bytes(b"fixture bubblewrap binary")
        self.identity = {
            "execution_backend": "merlin", "run_id": "fixture-run",
            "trial_id": "fixture-trial", "hostname": "fixture-host",
        }
        self.environment = {
            "RH2_JOB_RUN_ID": "fixture-run", "ARNOLD_TRIAL_ID": "fixture-trial",
            "RH2_CHECKPOINT_HDFS": "hdfs://fixture/checkpoints/fixture-run",
            "ARNOLD_APPLIED_GPU_NUM": "0",
        }
        self.commands = []
        self.binary_missing = False
        self.failure_stage = None
        self.effective_uid = 0
        self.sudo_available = True

    def which(self, name, *, path):
        self.assertEqual(path, "/usr/bin:/bin")
        if name == "bwrap":
            return None if self.binary_missing else str(self.binary)
        if name == "sudo" and not self.sudo_available:
            return None
        return "/usr/bin/" + name

    def fake_run(self, stage, command, environment, work, output, store, report, closed_outputs):
        self.commands.append((stage, command))
        if stage == self.failure_stage:
            raise RuntimeError("fixture command failed: " + stage)
        if stage == "cuga_bubblewrap_install":
            self.assertEqual(environment["DEBIAN_FRONTEND"], "noninteractive")
            self.binary_missing = False
        logs = {
            "cuga_bubblewrap_version": "bubblewrap fixture-version\n",
            "cuga_bubblewrap_package_identity":
                "bubblewrap\tfixture-version\tfixture-architecture\tinstall ok installed\n",
            "cuga_bubblewrap_package_files": str(self.binary) + "\n",
        }
        (output / (stage + ".log")).write_text(logs.get(stage, "fixture command completed\n"))

    def prepare(self, inputs=None):
        with mock.patch.object(bootstrap.socket, "gethostname", return_value="fixture-host"), \
             mock.patch.object(bootstrap.os, "geteuid", return_value=self.effective_uid), \
             mock.patch.object(bootstrap.shutil, "which", side_effect=self.which), \
             mock.patch.object(bootstrap, "run_command", side_effect=self.fake_run):
            return bootstrap.prepare_cuga_bubblewrap(
                inputs or {}, self.identity, self.environment, self.work, self.output,
                self.store, self.report,
            )

    def test_missing_cpu_binary_is_installed_and_actual_identity_is_recorded(self):
        self.binary_missing = True
        result = self.prepare()
        self.assertEqual([stage for stage, _ in self.commands], [
            "cuga_bubblewrap_package_index", "cuga_bubblewrap_install",
            "cuga_bubblewrap_version", "cuga_bubblewrap_package_identity",
            "cuga_bubblewrap_package_files",
        ])
        self.assertEqual(self.commands[0][1], ["/usr/bin/apt-get", "update"])
        self.assertEqual(self.commands[1][1][0], "/usr/bin/apt-get")
        self.assertEqual(self.commands[1][1][-1], "bubblewrap")
        self.assertEqual(result["package"]["version"], "fixture-version")
        self.assertEqual(result["sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(result["resolved_executable"], str(self.binary.resolve()))
        self.assertEqual(result["status"], "VERIFIED")
        self.assertTrue(result["installed_by_this_run"])

    def test_existing_binary_is_observed_without_package_mutation(self):
        result = self.prepare()
        self.assertFalse(result["installed_by_this_run"])
        self.assertFalse(any("install" in stage or "index" in stage for stage, _ in self.commands))

    def test_non_root_install_uses_noninteractive_system_sudo(self):
        self.effective_uid = 1000
        self.binary_missing = True
        result = self.prepare()
        self.assertEqual(self.commands[0][1], ["/usr/bin/sudo", "-n", "/usr/bin/apt-get", "update"])
        self.assertEqual(self.commands[1][1][:3], ["/usr/bin/sudo", "-n", "/usr/bin/apt-get"])
        self.assertTrue(result["installed_by_this_run"])

    def test_non_root_missing_sudo_fails_before_package_commands(self):
        self.effective_uid = 1000
        self.sudo_available = False
        self.binary_missing = True
        with self.assertRaisesRegex(PermissionError, "ROOT_OR_NONINTERACTIVE_SUDO_REQUIRED"):
            self.prepare()
        self.assertEqual(self.commands, [])
        self.assertTrue(self.binary_missing)

    def test_noninteractive_sudo_failure_propagates_without_install(self):
        self.effective_uid = 1000
        self.binary_missing = True
        self.failure_stage = "cuga_bubblewrap_package_index"
        with self.assertRaisesRegex(RuntimeError, "fixture command failed"):
            self.prepare()
        self.assertEqual(self.commands, [(
            "cuga_bubblewrap_package_index", ["/usr/bin/sudo", "-n", "/usr/bin/apt-get", "update"],
        )])
        self.assertTrue(self.binary_missing)
        self.assertFalse(self.report["bubblewrap"]["installed_by_this_run"])

    def test_existing_non_root_binary_needs_no_sudo(self):
        self.effective_uid = 1000
        self.sudo_available = False
        result = self.prepare()
        self.assertEqual(result["status"], "VERIFIED")
        self.assertFalse(result["installed_by_this_run"])
        self.assertFalse(any("install" in stage or "index" in stage for stage, _ in self.commands))

    def test_gpu_installs_only_cpu_measured_package_version(self):
        receipt = successful_cuga_binding(self.prepare())
        self.commands.clear()
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "1"
        self.binary_missing = True
        result = self.prepare({"cuga_binding_qualification": receipt})
        self.assertEqual(self.commands[1][1][-1], "bubblewrap=fixture-version")
        self.assertEqual(result["qualified_cpu_run_id"], "fixture-cpu-run")
        self.assertEqual(result["identity_mismatches"], [])

    def test_existing_gpu_binary_mismatch_fails_without_replacing_it(self):
        receipt = successful_cuga_binding(self.prepare())
        self.commands.clear()
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "1"
        changed = b"different existing bubblewrap"
        self.binary.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "DIFFERS_FROM_CPU_QUALIFICATION"):
            self.prepare({"cuga_binding_qualification": receipt})
        self.assertEqual(self.binary.read_bytes(), changed)
        self.assertFalse(any("install" in stage or "index" in stage for stage, _ in self.commands))
        observed = json.loads((self.output / "bubblewrap_identity.json").read_text())
        self.assertEqual(observed["status"], "IDENTITY_MISMATCH")
        self.assertIn("sha256", observed["identity_mismatches"])

    def test_gpu_package_identity_mismatch_is_rejected(self):
        receipt = successful_cuga_binding(self.prepare())
        receipt["bubblewrap"]["package"]["version"] = "different-package-version"
        self.commands.clear()
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "1"
        with self.assertRaisesRegex(ValueError, "DIFFERS_FROM_CPU_QUALIFICATION: package"):
            self.prepare({"cuga_binding_qualification": receipt})
        self.assertFalse(any("install" in stage or "index" in stage for stage, _ in self.commands))

    def test_missing_cpu_receipt_and_incomplete_sandbox_receipt_fail_before_commands(self):
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "1"
        with self.assertRaisesRegex(ValueError, "CPU_CUGA_BINDING_QUALIFICATION_REQUIRED"):
            self.prepare()
        self.assertEqual(self.commands, [])
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "0"
        receipt = successful_cuga_binding(self.prepare())
        self.commands.clear()
        self.environment["ARNOLD_APPLIED_GPU_NUM"] = "1"
        receipt["sandbox_probe"]["isolation_checks"]["procfs_absent"] = False
        with self.assertRaisesRegex(ValueError, "REAL_SANDBOX_QUALIFICATION_REQUIRED"):
            self.prepare({"cuga_binding_qualification": receipt})
        self.assertEqual(self.commands, [])

    def test_non_merlin_provisioning_is_rejected_before_commands(self):
        self.binary_missing = True
        self.environment["SLURM_JOB_ID"] = "fixture-slurm-job"
        with self.assertRaisesRegex(ValueError, "REQUIRES_REAL_MERLIN_ALLOCATION"):
            self.prepare()
        self.assertEqual(self.commands, [])

    def test_package_failure_propagates_without_running_later_commands(self):
        self.binary_missing = True
        self.failure_stage = "cuga_bubblewrap_install"
        with self.assertRaisesRegex(RuntimeError, "fixture command failed"):
            self.prepare()
        self.assertEqual([stage for stage, _ in self.commands], [
            "cuga_bubblewrap_package_index", "cuga_bubblewrap_install",
        ])
        self.assertEqual(self.report["bubblewrap"]["status"], "PREPARING")

    def test_failed_bubblewrap_preparation_precedes_cuga_bootstrap_or_restore(self):
        with mock.patch.object(
            bootstrap, "prepare_cuga_bubblewrap", side_effect=RuntimeError("fixture provisioning failed"),
        ), mock.patch.object(bootstrap.importlib, "import_module") as import_module, \
             mock.patch.object(bootstrap, "run_command") as run_command:
            with self.assertRaisesRegex(RuntimeError, "fixture provisioning failed"):
                bootstrap.prepare_cuga_context(
                    {}, self.identity, self.environment, self.work, self.output,
                    self.store, self.report,
                )
        import_module.assert_not_called()
        run_command.assert_not_called()


class CugaBindingSandboxTests(fixture.TemporaryArtifacts):
    def setUp(self):
        super().setUp()
        self.binary = self.work / "fixture-bwrap"
        self.binary.write_bytes(b"fixture sandbox binary")
        self.binding = {
            "python_base_prefix": self.work / "python-base",
            "public_application_lib": self.work.with_name(self.work.name + "-public") / "Libs",
        }
        self.environment = {"python": sys.executable}
        self.report["bubblewrap"] = {
            "executable": str(self.binary), **bootstrap.file_identity(self.binary),
        }
        self.command = [str(self.binary), "--unshare-all", "--share-net", "--cap-drop", "ALL"]
        self.commands = []
        self.failure = False
        self.checks = {key: True for key in (
            "private_repository_hidden", "procfs_absent", "work_directory", "public_modules_only",
        )}

    def fake_run(self, stage, command, environment, work, output, store, report, closed_outputs):
        self.commands.append(command)
        self.assertEqual(environment["PYTHONPATH"], "/app")
        if stage == "cuga_public_import_command":
            with mock.patch.object(sys, "argv", [sys.executable, command[-1]]), \
                 mock.patch.object(sys, "path", list(sys.path)), \
                 mock.patch.object(
                     cuga_process, "_python_runtime",
                     return_value=(Path(sys.executable), self.work / "python", self.work / "python-base"),
                 ), mock.patch.object(
                     cuga_process, "_bubblewrap_command", return_value=tuple(self.command),
                 ) as builder:
                exec(binding_qualification.BUILD_SANDBOX_COMMAND, {})
            arguments = builder.call_args.kwargs
            self.assertEqual(arguments["protected_roots"], (self.work,))
            self.assertEqual(arguments["owner_root"], self.output)
            self.assertEqual(arguments["worker_arguments"][1], binding_qualification.PUBLIC_IMPORT_PROBE)
        else:
            self.assertEqual(command, self.command)
            report.setdefault("stages", {})[stage] = {
                "pid": 12345, "exit_code": 1 if self.failure else 0,
            }
            if self.failure:
                raise RuntimeError("fixture namespace setup failed")
            bootstrap.write_json(output / "public_import_sandbox/public_import.json", {
                "status": "PASS", "isolation_checks": self.checks,
            })

    def qualify(self):
        with mock.patch.object(bootstrap, "run_command", side_effect=self.fake_run):
            return binding_qualification.qualify_public_import(
                bootstrap, self.binding, self.environment, self.work,
                self.output, self.store, self.report,
            )

    def test_probe_executes_builder_command_and_records_actual_checks(self):
        result = self.qualify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.commands[-1], self.command)
        probe = self.report["sandbox_probe"]
        self.assertEqual(probe["exit_code"], 0)
        self.assertTrue(probe["sandbox_executed"])
        self.assertEqual(probe["isolation_checks"], self.checks)
        self.assertNotIn("namespaces_verified", probe)

    def test_namespace_failure_propagates_without_passing_import_receipt(self):
        self.failure = True
        with self.assertRaisesRegex(RuntimeError, "fixture namespace setup failed"):
            self.qualify()
        self.assertNotIn("public_import", self.report)
        self.assertEqual(self.report["sandbox_probe"]["exit_code"], 1)
        self.assertEqual(self.report["sandbox_probe"]["isolation_checks"], {})


class CugaPreparationTests(fixture.TemporaryArtifacts):
    def test_pinned_bootstrap_precedes_restore_and_only_public_sources_are_copied(self):
        private = self.work / "private"
        private.mkdir()
        inputs = cuga_inputs()
        inputs["cuga_prior_launcher"] = content_record(b"fixture pinned launcher")
        for name in bootstrap.CUGA_PUBLIC_FILES:
            path = private / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture source")
        (private / "private_receipt.json").write_text("{}")
        events = []
        identity = {"hostname": "fixture-host", "run_id": "fixture-run",
                    "trial_id": "fixture-trial", "checkpoint_path": "hdfs://fixture/run"}
        preparation = SimpleNamespace(
            LEGACY_REPOSITORY=self.work / "legacy-repository",
            LEGACY_WORK=self.work / "legacy-work",
        )

        def staging_prefix(path):
            self.assertEqual(path.read_bytes(), b"fixture pinned launcher")
            events.append("staging_prefix")
            return b"#!/bin/bash\nexit 0\n"

        def run(stage, command, environment, work, output, store, report, closed_outputs):
            self.assertEqual(stage, "cuga_runtime_bootstrap")
            self.assertEqual(command[0], "bash")
            self.assertEqual(Path(command[1]).read_bytes(), b"#!/bin/bash\nexit 0\n")
            events.append("bootstrap")

        def restore(origin, actual_identity, **kwargs):
            self.assertEqual(events, ["bubblewrap", "staging_prefix", "bootstrap"])
            self.assertIs(origin, inputs["cuga_runtime"])
            self.assertEqual(actual_identity["compatibility_execution_id"], "merlin_fixture-trial_cuga")
            events.append("restore")
            bound = {"hostname": actual_identity["hostname"], "python": "/fixture/cuga/bin/python"}
            kwargs["report"].update(environment=bound, runtime_identity={"base_prefix": "/fixture/python"})
            bootstrap.write_json(kwargs["environment_path"], bound)

        preparation.staging_prefix = staging_prefix
        preparation.restore_prepared_runtime = restore
        with mock.patch.object(
            bootstrap, "prepare_cuga_bubblewrap", side_effect=lambda *args: events.append("bubblewrap"),
        ), mock.patch.object(bootstrap.importlib, "import_module", return_value=preparation), \
             mock.patch.object(bootstrap, "run_command", side_effect=run), \
             mock.patch.object(
                 bootstrap, "HdfsStore",
                 side_effect=lambda prefix, work, output: fixture.ArtifactStore(work, output),
             ):
            binding = bootstrap.prepare_cuga_context(
                inputs, identity, {}, private, self.output,
                fixture.ArtifactStore(private, self.output), self.report,
            )
        self.assertEqual(events, ["bubblewrap", "staging_prefix", "bootstrap", "restore"])
        self.assertEqual(binding["repository_root"], private)
        self.assertEqual(binding["python_base_prefix"], Path("/fixture/python"))
        public = binding["public_application_lib"]
        actual = {str(path.relative_to(public.parent)) for path in public.rglob("*") if path.is_file()}
        self.assertEqual(actual, set(bootstrap.CUGA_PUBLIC_FILES))
        self.assertFalse((public / "private_receipt.json").exists())
        self.assertEqual(json.loads(binding["cuga_environment"].read_text())["hostname"], "fixture-host")

    def test_selector_uses_explicit_artifact_directory(self):
        private = self.work / "private"
        private.mkdir()
        selection = self.output / "selection"
        config = {"schema": "appworld_benchmark_v1", "development_execution": {
            "api_predictor": {"demo_task_ids": ["case_1"]},
            "methods": {"official_react_code": {}, "official_full_code": {},
                        "official_function_calling": {}},
        }}
        manifest = {
            "schema": "appworld_split_membership_v1", "appworld_version": "fixture",
            "data_root": str(private), "splits": {"train": {
                "task_ids": ["case_1", "case_2", "case_3"], "task_count": 3, "scenario_count": 1,
                "membership_file": str(private / "train.txt"), "membership_sha256": "a" * 64,
            }},
        }
        bootstrap.write_json(self.output / "bound_config.yaml", config)
        bootstrap.write_json(private / "split_manifest.json", manifest)
        result = subprocess.run(
            [sys.executable, "-c", bootstrap.SELECT_TASK, str(private),
             "train", "0", "41", str(selection), str(self.output)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((selection / "selection.json").read_text())["task_id"], "case_2")
        self.assertFalse((private / "artifacts").exists())


class CugaDispatchTests(fixture.TemporaryArtifacts):
    def test_existing_cuga_pipeline_receives_all_bindings_and_root_score_path(self):
        directory = self.output / "development"
        binding = {name: self.work / name for name in (
            "cuga_environment", "python_base_prefix", "public_application_lib", "repository_root",
        )}
        context = {
            "common": ["--config", str(self.work / "config.yaml")],
            "native_environment": self.work / "native.json",
            "split_manifest": self.work / "split.json",
            "qualification": self.work / "qualification",
            "environment": {}, "cuga": binding,
        }
        commands = []

        def run(stage, command, environment, work, output, store, report, closed_outputs):
            commands.append(command)
            self.assertFalse(directory.exists())
            directory.mkdir()
            bootstrap.write_json(directory / "pipeline.json", {"status": "SCORED_DEVELOPMENT_ATTEMPT"})
            bootstrap.write_json(directory / "development_score.json", {"native_tgc": False})

        with mock.patch.object(bootstrap, "run_command", side_effect=run):
            bootstrap.run_selected_attempt(
                bootstrap.resolve_development_attempt({"development_attempt": cuga_attempt()}),
                "fixture_task", context=context, model_python=sys.executable,
                work=self.work, output=self.output, store=self.store, report=self.report,
                directory=directory, stage="cuga_fixture",
            )
        command = commands[0]
        self.assertIn("population_dynamics.appworld_benchmark_cuga_development", command)
        self.assertNotIn("--agent", command)
        self.assertNotIn("--max-model-calls", command)
        for name, value in binding.items():
            self.assertEqual(command[command.index("--" + name.replace("_", "-")) + 1], str(value))


class CugaScoreTests(fixture.TemporaryArtifacts):
    def setUp(self):
        super().setUp()
        relative = "development/cuga/train/seed_41/eligible_0"
        self.directory = self.output / relative
        self.directory.mkdir(parents=True)
        self.row = {"output_directory": relative, "stage": "cuga_fixture", "agent": "cuga",
                    "task_id": "fixture_task", "split": "train", "seed": 41, "method": CUGA_METHOD}
        identity = {key: self.row[key] for key in ("task_id", "split", "seed", "method")}
        cleanup = {"child_reaped": True, "exit_code": 0, "termination_signals": [],
                   "request_outcome_unresolved": False}
        self.pipeline = {
            **identity, "schema": "appworld_cuga_development_pipeline_v1", "formal_eligible": False,
            "status": "SCORED_DEVELOPMENT_ATTEMPT", "native_tgc": False,
            "native_client": {"status": "closed"},
            "model_server": {"status": "STOPPED", "server_cleanup": {"server_parent_reaped": True}},
            "processes": {"scorer": {"exit_code": 0, "cleanup": {"parent_reaped": True}}},
            "cuga_owner": {
                "status": "completed", "rpc_outcome_unresolved": False,
                "worker": {"reaped": True, "exit_code": 0, "termination_signals": []},
                "relay": {"status": "closed", "rpc_outcome_unresolved": False,
                          "native_cleanup": cleanup},
            },
        }
        self.execution = {
            **identity, "schema": "appworld_development_execution_v1", "formal_eligible": False,
            "run_root": str(self.directory / "native_run"), "terminal": True,
            "rpc_outcome_unresolved": False, "status": "completed",
            "native_cleanup": copy.deepcopy(cleanup), "usage_complete": False,
            "cuga_usage_transport_coverage_qualified": False,
            "cuga_model_usage": {"fixture": "unqualified callback observation"},
        }
        self.score = {**identity, "schema": "appworld_development_score_v1", "formal_eligible": False,
                      "scoring_status": "completed", "native_tgc": False}
        self.report["stages"] = {"cuga_fixture": {
            "exit_code": 0, "outputs_closed": True, "cleanup": {"owner_terminated": True},
        }}
        self.report["decisive_artifacts"] = {
            relative + "/" + name + ".json": {}
            for name in ("pipeline", "execution_receipt", "development_score")
        }
        self.write_records()

    def write_records(self):
        for name, value in (("pipeline", self.pipeline), ("execution_receipt", self.execution),
                            ("development_score", self.score)):
            bootstrap.write_json(self.directory / (name + ".json"), value)

    def test_negative_score_retains_incomplete_cost_coverage(self):
        result = bootstrap.batch_attempt_score(self.row, self.output, self.report)
        self.assertIs(result["native_tgc"], False)
        self.assertIs(result["usage_complete"], False)
        self.assertIs(result["cuga_usage_transport_coverage_qualified"], False)
        self.assertIsNone(result["lm_call_count"])
        self.assertEqual(result["cuga_model_usage"], self.execution["cuga_model_usage"])
        self.assertTrue(result["closed_outputs_persisted"])

    def test_unknown_native_execution_is_not_a_negative_score(self):
        self.execution["rpc_outcome_unresolved"] = True
        self.write_records()
        with self.assertRaisesRegex(ValueError, "COMPLETED_SCORE_REQUIRED"):
            bootstrap.batch_attempt_score(self.row, self.output, self.report)

    def test_unresolved_process_cleanup_is_rejected(self):
        for component in ("server", "native", "worker", "scorer"):
            original = copy.deepcopy(self.pipeline)
            if component == "server":
                self.pipeline["model_server"]["status"] = "UNKNOWN"
            elif component == "native":
                self.pipeline["native_client"]["status"] = "cleanup_unresolved"
            elif component == "worker":
                self.pipeline["cuga_owner"]["worker"]["reaped"] = False
            else:
                self.pipeline["processes"]["scorer"]["cleanup"]["parent_reaped"] = False
            self.write_records()
            with self.subTest(component=component), self.assertRaisesRegex(
                ValueError, "PROCESS_CLOSURE_UNCONFIRMED",
            ):
                bootstrap.batch_attempt_score(self.row, self.output, self.report)
            self.pipeline = original

    def test_changed_identity_and_missing_publication_are_rejected(self):
        self.score["seed"] = 43
        self.write_records()
        with self.assertRaisesRegex(ValueError, "IDENTITY_DIFFERS"):
            bootstrap.batch_attempt_score(self.row, self.output, self.report)
        self.score["seed"] = 41
        self.write_records()
        self.report["decisive_artifacts"].clear()
        with self.assertRaisesRegex(ValueError, "NOT_PERSISTED"):
            bootstrap.batch_attempt_score(self.row, self.output, self.report)


class NativeSessionDataLinkTests(fixture.TemporaryArtifacts):
    def setUp(self):
        super().setUp()
        self.data_root = self.work / "admitted"
        self.data_directory = self.data_root / "data"
        self.data_directory.mkdir(parents=True)
        (self.data_directory / "input_only.txt").write_text("private native input")
        self.native_root = self.output / "development/cuga/train/seed_41/eligible_0/native_run"
        self.native_root.mkdir(parents=True)
        self.link = self.native_root / "data"
        self.link.symlink_to(self.data_directory, target_is_directory=True)
        self.relative_root = self.native_root.relative_to(self.output).as_posix()
        self.store.native_session_roots = (self.relative_root,)
        self.archive = {"uri": "hdfs://fixture/data.tar.gz", "sha256": "b" * 64}
        self.prepared = {
            "status": "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION",
            "exit_code": 0, "persisted": True, "data_root": str(self.data_root),
            "data_hdfs_uri": self.archive["uri"], "data_bundle_sha256": self.archive["sha256"],
            "environment": {"status": "PREPARED_RUNTIME_AND_SPLIT_METADATA",
                            "data_root": str(self.data_root)},
        }
        bootstrap.write_json(self.work / "native_preparation.receipt.json", self.prepared)
        bootstrap.write_json(self.output / "input_admission.json", {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED", "data_archive": self.archive,
        })
        self.session = {
            "schema": "appworld_native_tool_session_v1", "scope": "development_tools_only",
            "formal_eligible": False, "run_root": str(self.native_root), "data_root": str(self.data_root),
        }
        self.session_path = self.native_root / "native_session.json"
        bootstrap.write_json(self.session_path, self.session)

    def test_registered_session_without_canonical_execution_excludes_data(self):
        self.assertFalse((self.native_root.parent / "execution_receipt.json").exists())
        self.assertNotIn("native_data_directory", self.session)
        self.assertEqual(bootstrap.admitted_native_data_link(self.store, self.output, self.link),
                         {"target": str(self.data_directory), "data_archive": self.archive})
        records = bootstrap.publish_closed(self.store, self.output, [self.native_root])
        self.assertEqual(set(records), {self.relative_root + "/native_session.json"})
        self.assertFalse(any("input_only" in name for _, name in self.store.uploads))
        names = bootstrap.decisive_artifact_names(self.store)
        self.assertIn(self.relative_root.rsplit("/", 1)[0] + "/execution_receipt.json", names)

    def test_wrong_owner_schema_scope_data_or_target_is_rejected(self):
        for key, value in (("run_root", str(self.work)), ("data_root", str(self.work)),
                           ("schema", "other"), ("scope", "formal"), ("formal_eligible", 0)):
            bootstrap.write_json(self.session_path, {**self.session, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER"):
                bootstrap.admitted_native_data_link(self.store, self.output, self.link)
        bootstrap.write_json(self.session_path, self.session)
        self.link.unlink()
        self.link.symlink_to(self.work, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)

    def test_unregistered_link_and_missing_or_linked_session_are_rejected(self):
        del self.store.native_session_roots
        with self.assertRaisesRegex(ValueError, "OUTPUT_SYMLINK"):
            bootstrap.publish_closed(self.store, self.output, [self.link])
        self.store.native_session_roots = (self.relative_root,)
        self.session_path.unlink()
        with self.assertRaisesRegex(ValueError, "ADMISSION_METADATA"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)
        receipt = self.work / "other_session.json"
        bootstrap.write_json(receipt, self.session)
        self.session_path.symlink_to(receipt)
        with self.assertRaisesRegex(ValueError, "ADMISSION_METADATA"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)

    def test_original_archive_binding_and_generic_execution_contract_stay_required(self):
        bootstrap.write_json(self.output / "input_admission.json", {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED",
            "data_archive": {**self.archive, "sha256": "c" * 64},
        })
        with self.assertRaisesRegex(ValueError, "ADMISSION_IDENTITY"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)
        bootstrap.write_json(self.output / "input_admission.json", {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED", "data_archive": self.archive,
        })
        self.store.native_attempt_roots = (self.relative_root,)
        with self.assertRaisesRegex(ValueError, "ADMISSION_METADATA"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)
        execution = {"schema": "appworld_development_execution_v1", "run_root": str(self.native_root),
                     "data_root": str(self.data_root)}
        bootstrap.write_json(self.native_root / "execution_receipt.json", execution)
        with self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.link)
        bootstrap.write_json(self.native_root / "execution_receipt.json", {
            **execution, "native_data_directory": str(self.data_directory),
        })
        self.assertIsNotNone(bootstrap.admitted_native_data_link(self.store, self.output, self.link))

"""Test embedded launcher orchestration with bounded local fixtures and children."""

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


class _FastWait:
    """Exercise health polling promptly while preserving a real child process."""

    def __init__(self, child):
        self.child = child
        self.wait_count = 0

    def __getattr__(self, name):
        return getattr(self.child, name)

    def wait(self, timeout=None):
        self.wait_count += 1
        limit = 0.05 if self.wait_count == 1 else 5.0
        return self.child.wait(timeout=limit if timeout is None else min(timeout, limit))


class LfadsPriorLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launcher = (
            Path(__file__).resolve().parents[2]
            / "Output/neural_population_dynamics/operations/lfads_prior_refit.sbatch"
        )
        source = launcher.read_text(encoding="utf-8")
        cls.shell_prelude = source.split("export PYTEST_DISABLE_PLUGIN_AUTOLOAD=", 1)[0]
        embedded = source.split("<<'PY'\n", 1)[1]
        embedded = embedded.rsplit("\nPY", 1)[0]
        tree = ast.parse(embedded, filename=str(launcher))
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        ]
        if len(functions) != 1:
            raise AssertionError("Expected exactly one embedded execute()")
        cls.execute_code = compile(
            ast.Module(body=functions, type_ignores=[]), str(launcher), "exec"
        )
        for name in ("run_action", "validate_comparison_cache", "stage_cpu_inputs"):
            selected = [node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name]
            if len(selected) != 1:
                raise AssertionError("Expected exactly one embedded " + name + "()")
            setattr(cls, name + "_code", compile(
                ast.Module(body=selected, type_ignores=[]), str(launcher), "exec"))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lfads-launcher-test-")
        self.addCleanup(self.directory.cleanup)
        self.children = []
        self.addCleanup(self.cleanup_children)
        self.record = {"commands": []}
        self.saved = []
        self.namespace = {
            "work": Path(self.directory.name),
            "record": self.record,
            "save": lambda: self.saved.append(copy.deepcopy(self.record)),
            "health": mock.Mock(),
            "os": os,
            "signal": signal,
            "time": time,
            "subprocess": SimpleNamespace(
                Popen=self.start_child,
                STDOUT=subprocess.STDOUT,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        }
        exec(self.execute_code, self.namespace)
        self.execute = self.namespace["execute"]

    def start_child(self, *args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        self.children.append(child)
        return _FastWait(child)

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            child.wait(timeout=5)

    def assert_receipt(self, status, exit_code):
        self.assertEqual(len(self.record["commands"]), 1)
        entry = self.record["commands"][0]
        self.assertEqual(entry["status"], status)
        self.assertEqual(entry["exit_code"], exit_code)
        self.assertGreaterEqual(entry["elapsed_seconds"], 0)
        self.assertEqual(self.saved[-1]["commands"][0], entry)
        self.assertEqual(self.saved[0]["commands"][0]["status"], "RUNNING")
        return entry

    def test_successful_child_records_pass(self):
        command = [sys.executable, "-c", "print('completed')"]
        self.execute("success", command)
        entry = self.assert_receipt("PASS", 0)
        self.assertEqual(entry["command"], command)
        self.assertEqual(entry["pid"], self.children[0].pid)
        self.assertNotIn("interruption_type", entry)
        self.assertEqual(
            (Path(self.directory.name) / "success.log").read_text().strip(), "completed"
        )

    def test_nonzero_child_records_failure_before_raising(self):
        with self.assertRaisesRegex(RuntimeError, r"nonzero:EXIT_7"):
            self.execute("nonzero", [sys.executable, "-c", "raise SystemExit(7)"])
        self.assert_receipt("FAILED", 7)
        self.assertEqual(self.children[0].poll(), 7)

    def test_health_error_terminates_real_child_and_records_interruption(self):
        error = RuntimeError("resource health failed")
        self.namespace["health"] = mock.Mock(side_effect=[None, error])
        with self.assertRaisesRegex(RuntimeError, "resource health failed") as caught:
            self.execute(
                "interrupted", [sys.executable, "-c", "import time; time.sleep(60)"]
            )
        self.assertIs(caught.exception, error)
        entry = self.assert_receipt("INTERRUPTED", -signal.SIGTERM)
        self.assertEqual(entry["interruption_type"], "RuntimeError")
        self.assertEqual(entry["interruption_reason"], "resource health failed")
        self.assertEqual(entry["termination_signal"], "SIGTERM")
        self.assertEqual(self.namespace["health"].call_count, 2)
        self.assertEqual(self.children[0].poll(), -signal.SIGTERM)

    def test_live_child_records_termination_pending(self):
        child = mock.Mock(pid=424242)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("stubborn", 0.05)
        killpg = mock.Mock()
        self.namespace["os"] = SimpleNamespace(killpg=killpg)
        self.namespace["subprocess"].Popen = mock.Mock(return_value=child)
        self.namespace["health"] = mock.Mock(
            side_effect=[None, RuntimeError("resource health failed")]
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute("pending", ["mock-stubborn-child"])
        entry = self.assert_receipt("TERMINATION_PENDING", None)
        self.assertEqual(entry["interruption_type"], "RuntimeError")
        self.assertEqual(entry["interruption_reason"], "resource health failed")
        self.assertEqual(entry["termination_signal"], "SIGTERM")
        killpg.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=30)] * 2)
        self.assertEqual(self.children, [])


    def _configure_action(self, action="compare"):
        work = Path(self.directory.name)
        self.record["scientific_claim_qualified"] = False
        self.calls = []
        self.namespace.update(
            action=action, device="cpu", training=work/"training",
            acquisition=work/"acquisition", fields=work/"fields",
            migration_path=work/"migration.json", existing_cache=work/"source_cache",
            sys=sys, json=json, Path=Path, shutil=shutil,
            datetime=datetime, timezone=timezone,
            validate_comparison_cache=mock.Mock(return_value={"status": "VERIFIED"}),
            execute=self._fake_operation)
        exec(self.run_action_code, self.namespace)
        return self.namespace["run_action"]

    def _fake_operation(self, label, command):
        self.calls.append((label, command))
        if label == "posterior_fit":
            return
        output = Path(command[command.index("--output") + 1])
        if label.endswith("_forecast"):
            output.mkdir(parents=True)
            (output/"evaluation.json").write_text(json.dumps({"status": "EVALUATED_DEVELOPMENT"}))
        else:
            output.write_text(json.dumps({"status": "PASS"}))

    def test_comparison_forwards_identical_inputs_and_isolates_prior_arms(self):
        run_action = self._configure_action()
        run_action()
        self.namespace["validate_comparison_cache"].assert_called_once_with(
            self.namespace["existing_cache"])
        self.assertEqual([label for label, _ in self.calls], [
            "original_prior_forecast", "original_prior_replay",
            "fitted_prior_forecast", "fitted_prior_replay"])
        outputs = []
        for label, command in self.calls:
            self.assertNotIn("Libs.population_dynamics.lfads_prior_refit", command)
            for flag, name in (("--acquisition", "acquisition"), ("--fields", "fields"),
                               ("--training", "training"), ("--device", "device"),
                               ("--inference-migration", "migration_path")):
                self.assertEqual(command.count(flag), 1)
                self.assertEqual(command[command.index(flag)+1], str(self.namespace[name]))
            output = Path(command[command.index("--output")+1])
            outputs.append(output)
            arm = "fitted_prior" if label.startswith("fitted") else "original_prior"
            self.assertEqual(output.parent, Path(self.directory.name)/arm)
            if arm == "fitted_prior":
                self.assertEqual(command.count("--prior-refit"), 1)
                self.assertEqual(command[command.index("--prior-refit")+1],
                                 str(self.namespace["existing_cache"]))
            else:
                self.assertNotIn("--prior-refit", command)
            if label.endswith("_forecast"):
                self.assertEqual(command[2], "Libs.population_dynamics.lfads_trajectory_evaluation")
                self.assertEqual(command[command.index("--config")+1],
                    str(Path(self.directory.name)/"Libs/configs/neural_population_matched_forecast.yaml"))
            else:
                self.assertEqual(command[2], "Libs.population_dynamics.lfads_forecast_replay")
                self.assertEqual(command[command.index("--forecasts")+1],
                                 str(output.parent/"forecasts"))
        self.assertEqual(len(set(outputs)), 4)
        self.assertEqual(self.record["status"], "COMPARED_PRIORS_DEVELOPMENT")
        self.assertFalse(self.record["scientific_claim_qualified"])
        self.assertTrue(all(arm["status"] == "VERIFIED" for arm in self.record["comparison_arms"]))

    def test_fit_action_preserves_fresh_extraction_command(self):
        run_action = self._configure_action("fit")
        self.namespace["existing_cache"] = None
        run_action()
        self.assertEqual(self.calls, [("posterior_fit", [
            sys.executable, "-m", "Libs.population_dynamics.lfads_prior_refit",
            "--acquisition", str(self.namespace["acquisition"]),
            "--fields", str(self.namespace["fields"]),
            "--training", str(self.namespace["training"]),
            "--output", str(Path(self.directory.name)/"cache"),
            "--device", "cpu", "--inference-migration", str(self.namespace["migration_path"])])])
        self.namespace["validate_comparison_cache"].assert_not_called()
        self.assertEqual(self.record["status"], "FITTED_PRIOR_ONLY")

    def test_fit_action_preserves_existing_cache_resume(self):
        run_action = self._configure_action("fit")
        source = self.namespace["existing_cache"]
        source.mkdir()
        document = {"episodes": [{"case_id": "first"}, {"case_id": "second"}]}
        (source/"posterior_cache.json").write_text(json.dumps(document))
        run_action()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], "posterior_fit")
        self.assertEqual(self.calls[0][1][-1], "--continue-cache")
        self.assertEqual(json.loads((Path(self.directory.name)/"cache/posterior_cache.json").read_text()), document)
        self.assertEqual(self.record["continued_cache"],
                         {"source": str(source), "completed_episodes_at_start": 2})
        self.assertEqual(self.record["status"], "FITTED_PRIOR_ONLY")

    def test_shell_preserves_fit_interfaces_and_requires_explicit_cpu_comparison(self):
        arguments = ["training", "acquisition", "fields", "migration", "output", "cache"]
        cases = [(None, "cuda", [], 0), (None, "cpu", arguments[:5], 0),
                 ("fit", "cpu", arguments, 0), ("fit", "cuda", arguments[:5], 0),
                 ("compare", "cpu", arguments, 0), ("compare", "cpu", [], 2),
                 ("compare", "cpu", arguments[:5], 2), ("compare", "cuda", arguments, 2),
                 ("unknown", "cpu", arguments, 2)]
        for action, device, args, expected in cases:
            with self.subTest(action=action, device=device, arguments=len(args)):
                env = dict(os.environ, NRNSIM_LFADS_DEVICE=device,
                           SLURMD_NODENAME="cpu04", SLURM_JOB_GPUS="")
                env.pop("NRNSIM_LFADS_ACTION", None)
                if action is not None:
                    env["NRNSIM_LFADS_ACTION"] = action
                result = subprocess.run(["bash", "-s", "--", *args], input=self.shell_prelude,
                                        text=True, capture_output=True, env=env, check=False)
                self.assertEqual(result.returncode, expected, result.stderr)

    def _configure_cache_validation(self):
        run_action = self._configure_action()
        cases = [{"case_id": f"train_block_{block:03d}_familiar_{template}"}
                 for block in range(14) for template in range(4)]
        run = {"binding": {"seed": 41}}
        cohort = SimpleNamespace(binding={"cohort": "fitting"})
        selected = {"training_updates": 40}
        migration = {"status": "qualified"}
        document = dict(model_parameters_updated=False, frozen_global_variables_verified=20,
                        training_updates_before_and_after=40, episodes=cases,
                        binding={"inference_migration": migration})
        cache = self.namespace["existing_cache"]
        cache.mkdir()
        (cache/"posterior_cache.json").write_text(json.dumps(document))
        self.namespace.update(
            run=run, _verified_training=mock.Mock(return_value=(cohort, run, None, selected, migration)),
            fitting_cases=mock.Mock(return_value=cases),
            load_prior_parameters=mock.Mock(return_value=({}, {"fitting_episodes": 56})),
            sha256_file=mock.Mock(return_value="run-digest"),
            _cache_identity=mock.Mock(side_effect=lambda binding, case: case),
            _cached_paths=mock.Mock())
        exec(self.validate_comparison_cache_code, self.namespace)
        return run_action, document

    def test_comparison_validates_every_complete_cached_episode_before_execution(self):
        run_action, document = self._configure_cache_validation()
        run_action()
        self.assertEqual(self.namespace["_cached_paths"].call_count, 56)
        self.assertEqual([call.args[1] for call in self.namespace["_cached_paths"].call_args_list],
                         document["episodes"])
        self.assertEqual(self.record["comparison_cache"]["verified_posterior_episodes"], 56)
        self.namespace["load_prior_parameters"].assert_called_once()

    def test_comparison_rejects_missing_or_changed_frozen_network_evidence(self):
        run_action, document = self._configure_cache_validation()
        for key, value in (("frozen_global_variables_verified", None),
                           ("frozen_global_variables_verified", 0),
                           ("frozen_global_variables_verified", True),
                           ("training_updates_before_and_after", None),
                           ("training_updates_before_and_after", 41),
                           ("model_parameters_updated", True)):
            with self.subTest(key=key, value=value):
                changed = dict(document)
                if value is None:
                    changed.pop(key)
                else:
                    changed[key] = value
                (self.namespace["existing_cache"]/"posterior_cache.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "COMPLETE_FROZEN_NETWORK_EXTRACTION_EVIDENCE_REQUIRED"):
                    run_action()
                self.assertEqual(self.calls, [])
                self.namespace["load_prior_parameters"].assert_not_called()

    def test_comparison_stops_before_either_arm_when_complete_prior_loader_rejects(self):
        run_action, _ = self._configure_cache_validation()
        self.namespace["load_prior_parameters"].side_effect = ValueError(
            "PRIOR_REFIT_NETWORK_OR_FITTING_COHORT_BINDING_CHANGED")
        with self.assertRaisesRegex(ValueError, "PRIOR_REFIT_NETWORK_OR_FITTING_COHORT_BINDING_CHANGED"):
            run_action()
        self.assertEqual(self.calls, [])
        self.namespace["_cached_paths"].assert_not_called()

    def test_comparison_stops_before_either_arm_when_cached_path_is_invalid(self):
        run_action, _ = self._configure_cache_validation()
        self.namespace["_cached_paths"].side_effect = ValueError("FITTING_POSTERIOR_CACHE_IDENTITY_CHANGED")
        with self.assertRaisesRegex(ValueError, "FITTING_POSTERIOR_CACHE_IDENTITY_CHANGED"):
            run_action()
        self.assertEqual(self.calls, [])

    def test_comparison_preserves_forecasts_if_replay_fails(self):
        run_action = self._configure_action()
        def failed_replay(label, command):
            if label == "original_prior_replay":
                raise RuntimeError("original_prior_replay:EXIT_1")
            self._fake_operation(label, command)
        self.namespace["execute"] = failed_replay
        with self.assertRaisesRegex(RuntimeError, "original_prior_replay:EXIT_1"):
            run_action()
        self.assertTrue((Path(self.directory.name)/"original_prior/forecasts/evaluation.json").is_file())
        self.assertFalse((Path(self.directory.name)/"fitted_prior").exists())
        self.assertNotEqual(self.record["status"], "COMPARED_PRIORS_DEVELOPMENT")

    def _cpu_staging_archive(self, name="archive", *, declare_aliases=True, distinct_checkpoints=False):
        archive = Path(self.directory.name)/name
        entries = []
        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        def add(relative, content="fixture"):
            path = archive/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            entries.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path)))
        run = {"best_checkpoint": "checkpoints/epoch", "latest_checkpoint":
               "checkpoints/later_epoch" if distinct_checkpoints else "checkpoints/epoch"}
        add("training/run.json", json.dumps(run))
        add("training/epochs.json", "{}")
        for checkpoint in sorted({run["best_checkpoint"], run["latest_checkpoint"]}):
            for name in ("model.json", "training.json", "weights.npz"):
                add("training/"+checkpoint+"/"+name, checkpoint+":"+name)
        for name in ("best", "latest"):
            link = run[name+"_checkpoint"]
            (archive/"training"/name).symlink_to(link)
            if declare_aliases:
                entries.append(dict(path="training/"+name, link_target=link))
        for name in ("design.json", "acquisition.json"):
            add("train/"+name)
        for name in ("dataset.json", "episodes.csv"):
            add("fields/"+name)
        for block in range(18):
            for template in range(4):
                case = f"train_block_{block:03d}_familiar_{template}"
                for root, names in (("train", ("report.json", "spikes.npz", "initial_parameters.npz")),
                                    ("fields", ("summary.json", "spatial_fields.npz"))):
                    for name in names:
                        add(root+"/"+case+"/"+name)
                if block < 14:
                    add("cache/"+case+".npz")
        for name in ("posterior_cache.json", "prior_parameters.npz"):
            add("cache/"+name)
        for name in ("inference_migration.json", "generation_memory_cpu_equivalence.json",
                     "full_population_cuda/replay.json"):
            add("migration/"+name)
        (archive/"delivery.json").write_text(json.dumps({"files": entries}))
        return archive, run, digest

    def _cpu_staging_namespace(self, archive, run, digest, *, action="fit", name="staging"):
        work = Path(self.directory.name)/name
        work.mkdir()
        record = {}
        namespace = dict(self.namespace, work=work, record=record, save=lambda: None,
            run=run, training=archive/"training", acquisition=archive/"train", fields=archive/"fields",
            migration_path=archive/"migration/inference_migration.json", existing_cache=archive/"cache",
            action=action, Path=Path, json=json, shutil=shutil, sha256_file=digest)
        exec(self.stage_cpu_inputs_code, namespace)
        return work, record, namespace

    def test_cpu_staging_adds_selection_payloads_and_fitted_cache_only_for_comparison(self):
        archive, run, digest = self._cpu_staging_archive()
        for action in ("fit", "compare"):
            with self.subTest(action=action):
                work, record, namespace = self._cpu_staging_namespace(
                    archive, run, digest, action=action, name=action)
                namespace["stage_cpu_inputs"]()
                self.assertEqual(record["staged_inputs"]["status"], "VERIFIED")
                self.assertEqual(len(record["staged_inputs"]["source_deliveries"]), 1)
                for block in range(18):
                    for template in range(4):
                        case = f"train_block_{block:03d}_familiar_{template}"
                        expected = block < 14 or action == "compare"
                        self.assertEqual((work/"inputs/train"/case/"spikes.npz").is_file(), expected)
                        self.assertEqual((work/"inputs/train_fields"/case/"spatial_fields.npz").is_file(), expected)
                        # Forecast/replay use NativeRenewalEpisode for selection
                        # cases, whose physical refractory check consumes this file.
                        parameters = work/"inputs/train"/case/"initial_parameters.npz"
                        self.assertEqual(parameters.is_file(), action == "compare" and block >= 14)
                        if parameters.is_file():
                            self.assertEqual(digest(parameters), digest(archive/"train"/case/parameters.name))
                self.assertEqual((work/"inputs/prior_refit/prior_parameters.npz").is_file(), action == "compare")
                if action == "compare":
                    for block in range(14):
                        for template in range(4):
                            self.assertTrue((work/"inputs/prior_refit"/
                                f"train_block_{block:03d}_familiar_{template}.npz").is_file())

        # Admission must bind the newly staged physical parameters to the source
        # archive, not merely copy a file with the expected name.
        parameters = archive/"train/train_block_014_familiar_0/initial_parameters.npz"
        parameters.write_text("changed source parameters")
        invalid_work = Path(self.directory.name)/"changed_parameters"
        invalid_work.mkdir()
        namespace.update(work=invalid_work, record={})
        with self.assertRaisesRegex(ValueError, r"ARCHIVE_INPUT_CHANGED:.*initial_parameters\.npz"):
            namespace["stage_cpu_inputs"]()



    def test_cpu_staging_rebuilds_unlisted_distinct_checkpoint_aliases(self):
        archive, run, digest = self._cpu_staging_archive(
            declare_aliases=False, distinct_checkpoints=True)
        work, record, namespace = self._cpu_staging_namespace(archive, run, digest)
        namespace["stage_cpu_inputs"]()
        staged = record["staged_inputs"]
        self.assertEqual(staged["status"], "VERIFIED")
        self.assertEqual(staged["planned_files"], staged["staged_files"])
        self.assertEqual(staged["staged_files"], len(staged["files"]))
        for alias in ("best", "latest"):
            destination = work/"inputs/training"/alias
            self.assertTrue(destination.is_symlink())
            self.assertEqual(str(destination.readlink()), run[alias+"_checkpoint"])
            for name in ("model.json", "training.json", "weights.npz"):
                self.assertEqual(digest(destination/name), digest(archive/"training"/run[alias+"_checkpoint"]/name))
            row = next(row for row in staged["files"] if row["path"] == "inputs/training/"+alias)
            self.assertFalse(row["alias_declared"])
            self.assertEqual(row["delivery"], str(archive/"delivery.json"))
            self.assertEqual(row["run_json_path"], "inputs/training/run.json")
            self.assertEqual(row["run_json_sha256"], digest(archive/"training/run.json"))
        self.assertNotEqual(str((work/"inputs/training/best").readlink()),
                            str((work/"inputs/training/latest").readlink()))

    def test_cpu_staging_rejects_alias_and_checkpoint_binding_conflicts(self):
        cases = (
            ("source_best", "TRAINING_CHECKPOINT_ALIAS_CHANGED"),
            ("source_latest", "TRAINING_CHECKPOINT_ALIAS_CHANGED"),
            ("absolute_alias", "TRAINING_CHECKPOINT_ALIAS_CHANGED"),
            ("declared_alias", "ARCHIVE_CHECKPOINT_ALIAS_CONFLICT"),
            ("declared_regular_file", "ARCHIVE_CHECKPOINT_ALIAS_CONFLICT"),
            ("unlisted_payload", "INPUT_NOT_DECLARED_BY_ARCHIVE"),
            ("changed_payload", "ARCHIVE_INPUT_CHANGED"),
            ("changed_run", "ARCHIVE_INPUT_CHANGED"),
        )
        for change, message in cases:
            with self.subTest(change=change):
                archive, run, digest = self._cpu_staging_archive(
                    name="archive_"+change, distinct_checkpoints=True)
                manifest = archive/"delivery.json"
                document = json.loads(manifest.read_text())
                if change in ("source_best", "source_latest", "absolute_alias"):
                    alias = "latest" if change == "source_latest" else "best"
                    path = archive/"training"/alias
                    path.unlink()
                    link = (str(archive/"training"/run["best_checkpoint"]) if change == "absolute_alias"
                            else run[("best" if alias == "latest" else "latest")+"_checkpoint"])
                    path.symlink_to(link)
                elif change in ("declared_alias", "declared_regular_file"):
                    row = next(row for row in document["files"] if row["path"] == "training/best")
                    if change == "declared_alias":
                        row["link_target"] = run["latest_checkpoint"]
                    else:
                        row.pop("link_target")
                        row.update(bytes=0, sha256="0"*64)
                elif change == "unlisted_payload":
                    path = "training/"+run["latest_checkpoint"]+"/weights.npz"
                    document["files"] = [row for row in document["files"] if row["path"] != path]
                elif change == "changed_payload":
                    (archive/"training"/run["latest_checkpoint"]/"weights.npz").write_text("changed")
                else:
                    (archive/"training/run.json").write_text(json.dumps(dict(run, unexpected=True)))
                manifest.write_text(json.dumps(document))
                work, record, namespace = self._cpu_staging_namespace(
                    archive, run, digest, name="stage_"+change)
                with self.assertRaisesRegex(ValueError, message):
                    namespace["stage_cpu_inputs"]()
                self.assertNotEqual(record.get("staged_inputs", {}).get("status"), "VERIFIED")
                if change in ("unlisted_payload", "changed_payload", "changed_run"):
                    self.assertFalse((work/"inputs/training/best").is_symlink())
                    self.assertFalse((work/"inputs/training/latest").is_symlink())


if __name__ == "__main__":
    unittest.main()

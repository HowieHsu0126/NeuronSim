"""Small archive/CLI regressions; payloads are synthetic bytes, not model arrays."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_lfads_conditional_forecast.py"
_spec = importlib.util.spec_from_file_location("lfads_prior_forecast_runner_under_test", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUNNER)


def _put(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _completed_forecast_parent(base, evaluator):
    parent = base / "parent"
    payload = _put(parent, "candidate/forecasts/origin.npz", b"synthetic forecast")
    scores = _put(parent, "candidate/forecasts/scores.csv", b"synthetic scores")
    evaluation = dict(status="EVALUATED_DEVELOPMENT", completed_origins=48, score_rows=144,
        evaluator_implementation_sha256=RUNNER.digest(evaluator),
        scores_sha256=scores["sha256"], artifacts={"origin.npz": payload["sha256"]},
        elapsed_seconds=911.77, forecast_continuation={"parent_job_id": "previous", "inherited_origins": 32})
    phase = dict(status="PASS", exit_code=0, command_started=True,
                 started_utc="source-start", finished_utc="source-end")
    rows = [payload, scores,
        _put(parent, "candidate/forecasts/evaluation.json", _json_bytes(evaluation)),
        _put(parent, "phases/forecast/receipt.json", _json_bytes(phase)),
        _put(parent, "candidate/replay.json", _json_bytes(dict(status="RUNNING", verified_origins=47)))]
    delivery = dict(status="RESOURCE_BLOCKED", exit_code=3, job_id="stopped", node="cpu04",
                    software=str(evaluator.parents[2]), files=rows)
    _put(parent, "delivery.json", _json_bytes(delivery))
    return parent


class FitArchive:
    """A producer-shaped receipt with distinct, declared selection payloads."""

    def __init__(self, base):
        self.root = base / "fit"
        self.source = base / "source"
        self.source_delivery = self.source / "delivery.json"
        self.config = {
            "fit_blocks": list(range(14)),
            "excluded_selection_blocks": list(range(14, 18)),
            "complete_episodes": 56,
            "posterior_draws_per_episode": 16,
        }
        self.cases = [f"train_block_{block:03d}_familiar_{template}"
                      for block in range(14) for template in range(4)]
        self.entries = {}
        self.initial_plan = []
        self.selection_files = {}
        source_rows = []
        for block in range(14, 18):
            for template in range(4):
                case = f"train_block_{block:03d}_familiar_{template}"
                for folder, anchor, additions in (
                    ("train", "report.json", ("spikes.npz", "initial_parameters.npz")),
                    ("train_fields", "summary.json", ("spatial_fields.npz",)),
                ):
                    for filename in (anchor, *additions):
                        relative = f"{folder}/{case}/{filename}"
                        row = _put(self.source, relative, ("fixture:" + relative).encode())
                        source_rows.append({"path": row["path"], "sha256": row["sha256"],
                                            "source_bytes": row["bytes"], "destination_bytes": row["bytes"]})
                        staged = dict(row, source=str(self.source / relative),
                                      path="inputs/" + relative, delivery=str(self.source_delivery))
                        if filename == anchor:
                            self.initial_plan.append(staged)
                        else:
                            self.selection_files[staged["path"]] = staged
        _put(self.source, "delivery.json", _json_bytes({"files": source_rows}))
        config_name = "Libs/configs/neural_population_lfads_prior_refit.yaml"
        self.entries[config_name] = _put(self.root, config_name, _json_bytes(self.config))
        for name in ["prior_parameters.npz", *[case + ".npz" for case in self.cases]]:
            relative = "cache/" + name
            self.entries[relative] = _put(self.root, relative, ("synthetic:" + name).encode())
        self.cache = {
            "status": "FITTED_FROZEN_NETWORK_AR_PRIOR",
            "fitting_cohort_extracted": True,
            "model_parameters_updated": False,
            "formal_test_accessed": False,
            "complete_episodes": 56,
            "posterior_paths": 896,
            "frozen_global_variables_verified": 152,
            "training_updates_before_and_after": 1120,
            "binding": {"case_ids": self.cases, "training_seed": 41,
                        "inference_migration": {"fixture": "pinned"}},
            "episodes": [{"case_id": case, "path": case + ".npz"} for case in self.cases],
        }
        self.reference = {
            "status": "FITTED_PRIOR_ONLY", "action": "fit", "training_seed": 41,
            "network_variant": "budget_20",
            "commands": [{"label": "posterior_fit", "status": "PASS", "exit_code": 0}],
            "staged_inputs": {
                "status": "VERIFIED", "files": self.initial_plan,
                "source_deliveries": {str(self.source_delivery): RUNNER.digest(self.source_delivery)},
            },
        }
        self.save_cache()

    def save_cache(self):
        name = "cache/posterior_cache.json"
        self.entries[name] = _put(self.root, name, _json_bytes(self.cache))
        self.reference["files"] = list(self.entries.values())
        _put(self.root, "delivery.json", _json_bytes(self.reference))


class StationaryInputPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lfads-prior-runner-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.archive = FitArchive(self.base)

    def plan(self, reference=None):
        return RUNNER.stationary_input_plan(
            self.archive.root, self.archive.reference if reference is None else reference)

    def test_completed_fit_preserves_plan_and_adds_declared_selection_and_complete_cache(self):
        before = copy.deepcopy(self.archive.reference)
        plan = self.plan()
        self.assertEqual(self.archive.reference, before)
        self.assertEqual(plan[:len(self.archive.initial_plan)], self.archive.initial_plan)
        indexed = {row["path"]: row for row in plan}
        self.assertEqual(len(indexed), len(plan))
        additions = {name: indexed[name] for name in self.archive.selection_files}
        self.assertEqual(additions, self.archive.selection_files)
        self.assertEqual(len(additions), 48)
        expected = {"inputs/prior_refit/" + name.removeprefix("cache/"): row
                    for name, row in self.archive.entries.items() if name.startswith("cache/")}
        actual = {row["path"]: row for row in plan if row["path"].startswith("inputs/prior_refit/")}
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(actual), 58)
        for name, row in actual.items():
            with self.subTest(path=name):
                original = expected[name]
                self.assertEqual(row["source"], str(self.archive.root / original["path"]))
                self.assertEqual((row["bytes"], row["sha256"]), (original["bytes"], original["sha256"]))

    def test_incomplete_or_unfrozen_cache_is_rejected_even_with_all_payload_files(self):
        original = copy.deepcopy(self.archive.cache)
        for changes in (
            {"status": "COMPLETE_FITTING_POSTERIOR_CACHE"},
            {"status": "COMPLETE_CACHE_PRIOR_FIT_FAILED"},
            {"model_parameters_updated": True},
            {"formal_test_accessed": True},
            {"frozen_global_variables_verified": True},
        ):
            with self.subTest(changes=changes):
                self.archive.cache = dict(copy.deepcopy(original), **changes)
                self.archive.save_cache()
                with self.assertRaisesRegex(ValueError, "COMPLETE_FROZEN_STATIONARY_CACHE_REQUIRED"):
                    self.plan()

    def test_failed_or_duplicate_posterior_command_cannot_be_hidden_by_fit_status(self):
        for commands in (
            [{"label": "posterior_fit", "status": "PASS", "exit_code": 1}],
            [{"label": "posterior_fit", "status": "INTERRUPTED", "exit_code": -15}],
            self.archive.reference["commands"] * 2,
        ):
            with self.subTest(commands=commands):
                reference = dict(self.archive.reference, commands=commands)
                with self.assertRaisesRegex(ValueError, "COMPLETED_STATIONARY_FIT_REQUIRED"):
                    self.plan(reference)

    def test_missing_episode_is_rejected_despite_claimed_complete_count(self):
        self.archive.cache["episodes"].pop()
        self.archive.save_cache()
        with self.assertRaisesRegex(ValueError, "COMPLETE_FITTING_CASES_REQUIRED"):
            self.plan()

    def test_episode_cannot_point_to_another_existing_case_payload(self):
        self.archive.cache["episodes"][0]["path"] = self.archive.cases[-1] + ".npz"
        self.archive.save_cache()
        with self.assertRaisesRegex(ValueError, "FITTING_CACHE_PATH_CHANGED"):
            self.plan()

    def test_cached_seed_must_match_the_completed_fit(self):
        self.archive.cache["binding"]["training_seed"] = 42
        self.archive.save_cache()
        with self.assertRaisesRegex(ValueError, "COMPLETE_FROZEN_STATIONARY_CACHE_REQUIRED"):
            self.plan()

    def test_changed_declared_cache_ledger_is_rejected_before_planning(self):
        path = self.archive.root / "cache/posterior_cache.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "ARCHIVED_INPUT_CHANGED"):
            self.plan()

    def test_selection_delivery_must_match_the_fit_record(self):
        path = self.archive.source_delivery
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SELECTION_SOURCE_DELIVERY_CHANGED"):
            self.plan()

    def test_intake_delegates_native_bindings_and_rejects_migration_mismatch(self):
        work = self.base / "work"
        run = {"binding": {"seed": 41}}
        _put(work, "inputs/training/run.json", _json_bytes(run))
        _put(work, "inputs/prior_refit/posterior_cache.json", _json_bytes(self.archive.cache))
        cohort = SimpleNamespace(binding={"cohort": "fixture"})
        selected = {"training_updates": 1120, "next_epoch": 20}
        migration = {"fixture": "pinned"}
        package = ModuleType("population_dynamics")
        package.__path__ = []
        prior = ModuleType("population_dynamics.lfads_prior_refit")
        prior.load_prior_parameters = mock.Mock(return_value=({}, {}))
        evaluation = ModuleType("population_dynamics.lfads_trajectory_evaluation")
        evaluation._verified_training = mock.Mock(return_value=(cohort, run, None, selected, migration))
        modules = {"population_dynamics": package, prior.__name__: prior, evaluation.__name__: evaluation}
        with mock.patch.dict(sys.modules, modules):
            proof = RUNNER.validate_stationary_inputs(work)
            self.assertEqual(proof["status"], "VERIFIED")
            prior.load_prior_parameters.assert_called_once_with(
                work / "inputs/prior_refit", run=run, selected=selected,
                cohort_binding=cohort.binding,
                training_run_sha256=RUNNER.digest(work / "inputs/training/run.json"))
            evaluation._verified_training.assert_called_once_with(
                work / "inputs/train", work / "inputs/train_fields", work / "inputs/training",
                work / "inputs/migration/inference_migration.json")
            migration["fixture"] = "different"
            with self.assertRaisesRegex(ValueError, "STATIONARY_PRIOR_INFERENCE_MIGRATION_CHANGED"):
                RUNNER.validate_stationary_inputs(work)




    def test_parent_intake_and_child_share_environment_before_native_imports(self):
        self._check_main_environment(replay_only=False)

    def test_replay_only_cli_stages_forecasts_without_launching_forecast_or_reusing_replay_progress(self):
        self._check_main_environment(replay_only=True)

    def _check_main_environment(self, replay_only):
        for name in ("neural_population_dynamics.yaml", "neural_population_lfads_trajectory.yaml",
                     "neural_population_matched_forecast.yaml"):
            relative = "Libs/configs/" + name
            self.archive.entries[relative] = _put(self.archive.root, relative, b"{}\n")
        self.archive.save_cache()
        software = self.base / "software"
        files = [_put(software, name, b"") for name in (
            "Libs/__init__.py", "Libs/population_dynamics/__init__.py",
            "Libs/population_dynamics/resource_probe.py")]
        _put(software, "delivery.json", _json_bytes(dict(status="ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS",
            validation_mode="consumers", requested_test_device="cpu", exit_code=0, files=files)))
        guard, output = self.base / "guard.py", self.base / "result"
        guard.write_text("required_memory_gib=1\n")
        package = ModuleType("population_dynamics")
        package.__path__ = []
        resources = ModuleType("population_dynamics.resource_probe")
        resources.probe = lambda *args, **kwargs: {"status": "PASS"}
        modules = {"population_dynamics": package, resources.__name__: resources}
        continuation_args = []
        if replay_only:
            evaluator = software / "Libs/population_dynamics/lfads_trajectory_evaluation.py"
            evaluator.write_text("# synthetic evaluator\n")
            parent = _completed_forecast_parent(self.base, evaluator)
            continuation = self.base / "continuation"
            continuation_files = [_put(continuation, "Libs/population_dynamics/" + name,
                evaluator.read_bytes() if name == evaluator.name else b"# synthetic continuation\n")
                for name in ("lfads_trajectory_evaluation.py", "lfads_forecast_continuation.py", "lfads_evaluator_compatibility.py")]
            _put(continuation, "delivery.json", _json_bytes(dict(status="PASS", exit_code=0, files=continuation_files)))
            continuation_args = ["--continue-from", str(parent), "--continuation-software", str(continuation), "--replay-only"]
        observed = []

        def observe(label, work):
            expected = dict(CUDA_VISIBLE_DEVICES="", NVIDIA_TF32_OVERRIDE="0",
                PYTHONPATH=str(work / "Libs"), TMPDIR=str(work / "scratch"),
                OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self.assertEqual({key: os.environ.get(key) for key in expected}, expected)
            observed.append((label, expected))

        def activate(work):
            observe("activate", work)

        def intake(work):
            # The empty input-plan mock replaces real staging, which creates
            # this parent before the native intake and continuation branches.
            (work / "inputs").mkdir(exist_ok=True)
            self.assertEqual([row[0] for row in observed], ["activate"])
            observe("intake", work)
            return dict(status="VERIFIED", complete_episodes=56,
                        frozen_global_variables_verified=152, training_updates_before_and_after=1120)

        def phase(command, *, cwd, env):
            self.assertEqual([row[0] for row in observed], ["activate", "intake"])
            self.assertEqual({key: env.get(key) for key in observed[-1][1]}, observed[-1][1])
            label = "replay" if replay_only else "forecast"
            self.assertEqual(command[3], label)
            if replay_only:
                self.assertIn("Libs.population_dynamics.lfads_forecast_replay", command)
                for name in ("evaluation.json", "scores.csv", "origin.npz"):
                    self.assertEqual((cwd / "candidate/forecasts" / name).read_bytes(),
                                     (parent / "candidate/forecasts" / name).read_bytes())
                self.assertFalse((cwd / "candidate/replay.json").exists())
            observed.append(("child", env))
            # Stop at the process boundary: no forecast, model or arrays run.
            return SimpleNamespace(returncode=3)

        argv = [str(RUNNER_PATH), "--fit-run", str(self.archive.root), "--software", str(software),
                "--guard", str(guard), "--output", str(output), *continuation_args]
        with mock.patch.object(RUNNER.sys, "argv", argv), mock.patch.dict(RUNNER.os.environ,
            {"SLURMD_NODENAME": "cpu03" if replay_only else "cpu04", "SLURM_JOB_ID": "fixture", "CUDA_VISIBLE_DEVICES": "must-be-hidden"},
            clear=True), mock.patch.dict(sys.modules, modules), \
            mock.patch.object(RUNNER, "activate_staged_software", side_effect=activate), \
            mock.patch.object(RUNNER, "stationary_input_plan", return_value=[]), \
            mock.patch.object(RUNNER, "validate_stationary_inputs", side_effect=intake), \
            mock.patch.object(RUNNER.shutil, "disk_usage", return_value=SimpleNamespace(free=64 * 1024**3)), \
            mock.patch.object(RUNNER.subprocess, "run", side_effect=phase), redirect_stdout(io.StringIO()):
            result = RUNNER.main()
            self.assertEqual(result, 3, (output / "delivery.json").read_text())
        self.assertEqual([row[0] for row in observed], ["activate", "intake", "child"])
        record = json.loads((output / "delivery.json").read_text())
        self.assertEqual(record["stationary_prior_intake"]["status"], "VERIFIED")
        self.assertEqual(record["status"], "RESOURCE_BLOCKED")
        self.assertEqual(len(record["commands"]), 1)
        if replay_only:
            self.assertTrue(record["replay_only"])
            self.assertEqual(record["reused_forecast"]["reused_origins"], 48)
            self.assertEqual(record["reused_forecast"]["forecast_origins_generated"], 0)
            self.assertEqual(record["reused_forecast"]["replay_progress_reused"], 0)
            self.assertEqual(json.loads((parent / "candidate/replay.json").read_text())["verified_origins"], 47)


class CompletedForecastReuseTests(unittest.TestCase):
    def test_replay_only_rejects_incomplete_or_changed_forecast_evidence(self):
        for change in ("evaluation_incomplete", "phase_incomplete", "phase_not_started", "changed_payload", "changed_scores_binding"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                evaluator = base / "software/Libs/population_dynamics/lfads_trajectory_evaluation.py"
                _put(base, str(evaluator.relative_to(base)), b"# fixture\n")
                parent = _completed_forecast_parent(base, evaluator)
                delivery = json.loads((parent / "delivery.json").read_text())
                if change == "changed_payload":
                    (parent / "candidate/forecasts/origin.npz").write_bytes(b"changed")
                else:
                    name = "candidate/forecasts/evaluation.json" if change in ("evaluation_incomplete", "changed_scores_binding") else "phases/forecast/receipt.json"
                    document = json.loads((parent / name).read_text())
                    if change == "evaluation_incomplete": document["status"] = "RUNNING"
                    elif change == "phase_incomplete": document.update(status="RESOURCE_BLOCKED", exit_code=-15)
                    elif change == "phase_not_started": document["command_started"] = False
                    else: document["scores_sha256"] = "0" * 64
                    replacement = _put(parent, name, _json_bytes(document))
                    delivery["files"] = [replacement if row["path"] == name else row for row in delivery["files"]]
                    _put(parent, "delivery.json", _json_bytes(delivery))
                with self.assertRaises(ValueError):
                    RUNNER.stage_completed_forecasts(parent, base / "candidate/forecasts")

    def test_cpu_placement_rejects_gpu_allocations_and_disallowed_nodes(self):
        cases = [("cpu03", "fixture", "", True), ("cpu04", "fixture", "", True),
                 ("cpu05", "fixture", "", False), ("gpu03", "fixture", "", False),
                 ("cpu03", "", "", False), ("cpu03", "fixture", "0", False)]
        for node, job, gpus, allowed in cases:
            with self.subTest(node=node, job=job, gpus=gpus), mock.patch.dict(RUNNER.os.environ,
                dict(SLURMD_NODENAME=node, SLURM_JOB_ID=job, SLURM_JOB_GPUS=gpus), clear=True):
                if allowed: RUNNER.validate_execution_allocation()
                else:
                    with self.assertRaisesRegex(ValueError, "AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED"):
                        RUNNER.validate_execution_allocation()


class StagedSoftwareImportTests(unittest.TestCase):
    def test_sparse_wrapper_uses_real_staged_intake_modules_and_rejects_conflicting_cache(self):
        with tempfile.TemporaryDirectory(prefix="lfads-staged-import-test-") as temporary:
            base = Path(temporary)
            control, work = base / "sparse_wrapper", base / "staged"
            # Match the sparse wrapper: it is a regular Libs package without
            # the external LFADS code supplied by the separate software archive.
            for name in ("Libs/__init__.py", "Libs/external/__init__.py",
                         "Libs/population_dynamics/__init__.py"):
                _put(control, name, b"")
            archived_runner = control / "Libs/scripts/run_lfads_conditional_forecast.py"
            archived_runner.parent.mkdir(parents=True)
            shutil.copy2(RUNNER_PATH, archived_runner)
            supplied = RUNNER_PATH.parents[2]
            _put(work, "Libs/__init__.py", (supplied / "Libs/__init__.py").read_bytes())
            # Copy the real supplied Python packages, including the original
            # LFADS implementation. No neural data, weights or arrays are used.
            for relative in ("Libs/population_dynamics", "Libs/external/lfads"):
                target = work / relative
                target.mkdir(parents=True, exist_ok=True)
                for source in (supplied / relative).glob("*.py"):
                    shutil.copy2(source, target / source.name)
            config_name = "Libs/configs/neural_population_lfads_trajectory.yaml"
            _put(work, config_name, (supplied / config_name).read_bytes())
            script = control / "Libs/scripts/import_intake.py"
            script.write_text(
                "import importlib, importlib.util, json, os, pathlib, sys, yaml\n"
                "runner_path, work, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]\n"
                "spec = importlib.util.spec_from_file_location('archived_runner', runner_path)\n"
                "runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)\n"
                "if mode.startswith('cached:'):\n"
                "    package = mode.split(':', 1)[1]\n"
                "    sys.path.insert(0, str(runner_path.parents[1]))\n"
                "    cached = importlib.import_module(package)\n"
                "    try: runner.activate_staged_software(work)\n"
                "    except ValueError as error:\n"
                "        assert str(error).startswith('CONFLICTING_STAGED_PACKAGE_IMPORT:')\n"
                "        assert sys.modules[package] is cached\n"
                "        print(json.dumps({'conflict_rejected': package})); sys.exit(0)\n"
                "    raise AssertionError('foreign cached package accepted')\n"
                "if mode == 'legacy': sys.path.insert(0, str(work / 'Libs'))\n"
                "else:\n"
                "    runner.configure_execution_environment(work)\n"
                "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and os.environ['NVIDIA_TF32_OVERRIDE'] == '0'\n"
                "    runner.activate_staged_software(work)\n"
                "try: prior = importlib.import_module('population_dynamics.lfads_prior_refit')\n"
                "except ModuleNotFoundError as error:\n"
                "    assert mode == 'legacy' and error.name == 'Libs.external.lfads', str(error)\n"
                "    print(json.dumps({'legacy_error': error.name})); sys.exit(0)\n"
                "assert mode == 'staged', 'legacy path unexpectedly found the staged external package'\n"
                "evaluation = importlib.import_module('population_dynamics.lfads_trajectory_evaluation')\n"
                "native = importlib.import_module('Libs.external.lfads.lfads')\n"
                "assert callable(prior.load_prior_parameters) and callable(evaluation._verified_training)\n"
                "modules = [prior, evaluation, native, sys.modules['Libs'], sys.modules['population_dynamics']]\n"
                "origins = {module.__name__: module.__file__ for module in modules}\n"
                "assert all(pathlib.Path(path).resolve().is_relative_to(work / 'Libs') for path in origins.values())\n"
                "training = importlib.import_module('population_dynamics.lfads_trajectory_training')\n"
                "config_path = work / 'Libs/configs/neural_population_lfads_trajectory.yaml'\n"
                "config_bytes = config_path.read_bytes(); config = yaml.safe_load(config_bytes)\n"
                "assert config['device'] == 'cuda', 'regression requires the frozen CUDA training configuration'\n"
                "os.environ.pop('NVIDIA_TF32_OVERRIDE', None)\n"
                "try: training.validate_config(config, 41)\n"
                "except ValueError as error: assert 'NVIDIA_TF32_OVERRIDE=0' in str(error), str(error)\n"
                "else: raise AssertionError('precision guard accepted missing override')\n"
                "child_env = runner.configure_execution_environment(work)\n"
                "training.validate_config(config, 41)\n"
                "assert child_env['NVIDIA_TF32_OVERRIDE'] == '0' and child_env['CUDA_VISIBLE_DEVICES'] == ''\n"
                "assert config_path.read_bytes() == config_bytes and yaml.safe_load(config_bytes) == config\n"
                "print(json.dumps({'staged_origins': origins, 'precision_guard_preserved': True}))\n")
            env = dict(os.environ, PYTHONPATH=str(control), PYTHONDONTWRITEBYTECODE="1",
                       CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            for mode in ("legacy", "staged", "cached:Libs", "cached:population_dynamics"):
                with self.subTest(mode=mode):
                    result = subprocess.run([sys.executable, "-B", str(script), str(archived_runner),
                                             str(work), mode], cwd=control, env=env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    record = json.loads(result.stdout.strip().splitlines()[-1])
                    if mode == "legacy":
                        self.assertEqual(record["legacy_error"], "Libs.external.lfads")
                    elif mode == "staged":
                        self.assertEqual(len(record["staged_origins"]), 5)
                        self.assertTrue(record["precision_guard_preserved"])
                    else:
                        self.assertEqual(record["conflict_rejected"], mode.split(":", 1)[1])


class PriorForecastCliTests(unittest.TestCase):
    def invoke(self, mode):
        argv = [str(RUNNER_PATH), *mode, "--software", "software", "--guard", "guard.py", "--output", "output"]
        # Stop valid modes at admission, before any filesystem or native work.
        with mock.patch.object(RUNNER.sys, "argv", argv), mock.patch.dict(
            RUNNER.os.environ, {"SLURMD_NODENAME": "not-an-authorized-node", "SLURM_JOB_ID": "fixture"},
            clear=True,
        ), redirect_stderr(io.StringIO()):
            return RUNNER.main()

    def test_source_modes_are_mutually_exclusive_and_required(self):
        for mode in ([], ["--fit-run", "fit", "--reference-run", "reference"]):
            with self.subTest(mode=mode), self.assertRaises(SystemExit) as caught:
                self.invoke(mode)
            self.assertEqual(caught.exception.code, 2)

    def test_mean_is_required_only_for_legacy_reference_mode(self):
        with self.assertRaisesRegex(ValueError, "STATIONARY_FORECAST_REQUIRES_COMPLETED_FIT"):
            self.invoke(["--reference-run", "reference"])

    def test_replay_only_requires_parent_before_node_admission(self):
        with self.assertRaisesRegex(ValueError, "REPLAY_ONLY_REQUIRES_COMPLETED_PARENT"):
            self.invoke(["--fit-run", "fit", "--replay-only"])

    def test_all_valid_mode_forms_reach_unchanged_resource_admission(self):
        for mode in (["--fit-run", "fit"], ["--reference-run", "reference", "--mean-run", "mean"],
                     ["--fit-run", "fit", "--mean-run", "mean"]):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED"
            ):
                self.invoke(mode)


if __name__ == "__main__":
    unittest.main()

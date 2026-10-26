"""Launcher contract tests using disposable inputs, not scientific evidence."""
import importlib.util
import json
import os
import shutil
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from .provenance import sha256_file

_ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = _ROOT / "Libs/scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("run_pfc_source_cases")
dataset = load_script("run_pfc_source_dataset")
packing = load_script("validate_pfc_source_packing")


class AllocationAdmissionReached(RuntimeError):
    """Stop after allocation admission, before any staging or source execution."""


class SourceAllocationTests(unittest.TestCase):
    def source_environment(self, **changes):
        values=dict(NEURONSIM_PROJECT_ROOT="/tmp", NEURONSIM_PFC_SOURCE_BACKEND="numpy",
            SLURM_JOB_ID="123", SLURMD_NODENAME="cpu03", SLURM_NTASKS="4",
            SLURM_CPUS_PER_TASK="1", NEURONSIM_PFC_SOURCE_CASES_JSON="/tmp/cases.json",
            NEURONSIM_PFC_SOURCE_CONFIG="/tmp/config.yaml",
            NEURONSIM_PFC_SOURCE_QUALIFICATION="/tmp/qualification")
        values.update(changes)
        return values

    def test_cpu03_and_existing_cpu_nodes_reach_scratch_after_admission(self):
        for node in ("cpu03", "cpu04", "cpu05", "cpu06"):
            for tasks in ("1", "4", "8"):
                with self.subTest(node=node, tasks=tasks), \
                     mock.patch.dict(os.environ, self.source_environment(
                         SLURMD_NODENAME=node, SLURM_NTASKS=tasks), clear=True), \
                     mock.patch("sys.argv", ["run_pfc_source_cases"]), \
                     mock.patch.object(launcher.Path, "mkdir",
                         side_effect=AllocationAdmissionReached) as create_work:
                    with self.assertRaises(AllocationAdmissionReached):
                        launcher.main()
                    create_work.assert_called_once_with(parents=True, exist_ok=False)

    def test_source_rejects_wrong_node_gpu_or_resource_geometry_before_staging(self):
        for changes in (dict(SLURMD_NODENAME="mgmt2"), dict(SLURMD_NODENAME="gpu03"),
                        dict(SLURMD_NODENAME="cpu02"), dict(SLURM_JOB_GPUS="0"), dict(CUDA_VISIBLE_DEVICES="0"),
                        dict(SLURM_NTASKS="0"), dict(SLURM_NTASKS="9"),
                        dict(SLURM_CPUS_PER_TASK="0"), dict(SLURM_CPUS_PER_TASK="2")):
            with self.subTest(changes=changes), \
                 mock.patch.dict(os.environ, self.source_environment(**changes), clear=True), \
                 mock.patch("sys.argv", ["run_pfc_source_cases"]), \
                 mock.patch.object(launcher.Path, "read_text") as read_config, \
                 mock.patch.object(launcher.Path, "mkdir") as create_work:
                with self.assertRaisesRegex(ValueError, "eight one-CPU tasks"):
                    launcher.main()
                read_config.assert_not_called()
                create_work.assert_not_called()

    def test_single_cuda_caller_preserves_existing_allocation_path(self):
        for node, partition in (("cpu03", "gpu"), ("cpu04", "cpu"), ("cpu05", "cpu"),
                                ("cpu06", "cpu"), ("gpu03", "gpu1")):
            env = self.source_environment(NEURONSIM_PFC_SOURCE_BACKEND="cuda",
                SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition, SLURM_JOB_GPUS="0,1",
                CUDA_VISIBLE_DEVICES="0,1", NEURONSIM_PFC_SOURCE_CASE_ID="case")
            env.pop("NEURONSIM_PFC_SOURCE_CASES_JSON")
            with self.subTest(node=node, partition=partition), \
                 mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch("sys.argv", ["run_pfc_source_cases"]), \
                 mock.patch.object(launcher.Path, "mkdir", side_effect=AllocationAdmissionReached):
                with self.assertRaises(AllocationAdmissionReached):
                    launcher.main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_cuda_packed_source_remains_rejected(self):
        env = self.source_environment(NEURONSIM_PFC_SOURCE_BACKEND="cuda",
            SLURMD_NODENAME="gpu03", SLURM_JOB_PARTITION="gpu1", SLURM_JOB_GPUS="0",
            CUDA_VISIBLE_DEVICES="0")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("sys.argv", ["run_pfc_source_cases"]):
            with self.assertRaisesRegex(ValueError, "NumPy backend"):
                launcher.main()

    def packing_arguments(self):
        return ["validate_pfc_source_packing", "--config", "/tmp/config.yaml",
                "--qualified-stream", "/tmp/qualification", "--native-validation", "/tmp/native",
                "--output", "/tmp/packing_output"]

    def test_supported_packing_geometry_reaches_output_after_admission(self):
        for node in ("cpu03", "cpu04", "cpu05", "cpu06"):
            for tasks in ("4", "8"):
                with self.subTest(node=node, tasks=tasks), \
                     mock.patch.dict(os.environ, self.source_environment(
                         SLURMD_NODENAME=node, SLURM_NTASKS=tasks), clear=True), \
                     mock.patch("sys.argv", self.packing_arguments()), \
                     mock.patch.object(packing.Path, "mkdir",
                         side_effect=AllocationAdmissionReached) as create_output:
                    with self.assertRaises(AllocationAdmissionReached):
                        packing.main()
                    create_output.assert_called_once_with(parents=True, exist_ok=False)

    def test_packing_rejects_wrong_node_gpu_or_unsupported_task_allocation(self):
        for changes in (dict(SLURMD_NODENAME="mgmt2"), dict(SLURMD_NODENAME="gpu03"),
                        dict(SLURMD_NODENAME="cpu02"), dict(SLURM_JOB_ID=""), dict(SLURM_JOB_GPUS="0"),
                        dict(CUDA_VISIBLE_DEVICES="0"), dict(SLURM_NTASKS="1"),
                        dict(SLURM_NTASKS="5"), dict(SLURM_CPUS_PER_TASK="2")):
            with self.subTest(changes=changes), \
                 mock.patch.dict(os.environ, self.source_environment(**changes), clear=True), \
                 mock.patch("sys.argv", self.packing_arguments()), \
                 mock.patch.object(packing.Path, "mkdir") as create_output:
                with self.assertRaisesRegex(ValueError, "four- or eight-task one-CPU"):
                    packing.main()
                create_output.assert_not_called()


class SourceLauncherTests(unittest.TestCase):
    def test_existing_single_array_and_packed_selection(self):
        config = dict(cases=[dict(case_id="a"), dict(case_id="b")])
        self.assertEqual(launcher.select_cases(config, explicit="b"), ["b"])
        self.assertEqual(launcher.select_cases(config, array_index="0"), ["a"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "cases.json"
            path.write_text('["b", "a"]')
            self.assertEqual(launcher.select_cases(config, case_file=path), ["b", "a"])
            self.assertEqual(launcher.select_cases(config, explicit="", array_index="", case_file=path), ["b", "a"])
            for values in ([], ["a", "a"], ["missing"], [1], ["../a"]):
                path.write_text(json.dumps(values))
                with self.assertRaises(ValueError):
                    launcher.select_cases(config, case_file=path)
        for kwargs in ({}, dict(explicit="a", array_index="0"), dict(array_index="-1")):
            with self.assertRaises(ValueError):
                launcher.select_cases(config, **kwargs)

    def test_failed_rank_stops_and_other_rank_retains_its_assignments(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            plan = dict(job_id="123", tasks=2, case_ids=["bad", "good", "not_run", "good_again"])
            launcher.write_json(work / "plan.json", plan)
            invoked = []
            def execute(work, case_id, rank, plan):
                invoked.append(case_id)
                return dict(case_id=case_id, rank=rank, status="FAILED" if case_id=="bad" else
                            "PFC_SOURCE_EXECUTION_COMPLETE", exit_code=int(case_id=="bad"))
            with mock.patch.object(launcher, "execute_case", side_effect=execute):
                for rank, expected in ((0, 1), (1, 0)):
                    with mock.patch.dict(os.environ, SLURM_PROCID=str(rank), SLURM_STEP_NUM_TASKS="2"):
                        self.assertEqual(launcher.run_rank(work), expected)
            self.assertEqual(invoked, ["bad", "good", "good_again"])
            failed = json.loads((work / "ranks/0.json").read_text())
            self.assertEqual([row["status"] for row in failed["cases"]], ["FAILED", "NOT_RUN"])
            self.assertIsNone(failed["cases"][1]["exit_code"])
            with mock.patch.dict(os.environ, SLURM_PROCID="0", SLURM_STEP_NUM_TASKS="3"):
                with self.assertRaisesRegex(ValueError, "geometry"):
                    launcher.run_rank(work)

    def test_eight_ranks_preserve_case_identity_and_failed_rank_tail(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            case_ids = ["case_%02d" % i for i in range(16)]
            launcher.write_json(work / "plan.json", dict(
                job_id="123", tasks=8, case_ids=case_ids))
            invoked = []
            def execute(work, case_id, rank, plan):
                invoked.append((case_id, rank))
                self.assertEqual(plan["case_ids"], case_ids)
                failed = case_id == case_ids[7]
                return dict(case_id=case_id, rank=rank, exit_code=int(failed),
                    status="FAILED" if failed else "PFC_SOURCE_EXECUTION_COMPLETE")
            with mock.patch.object(launcher, "execute_case", side_effect=execute):
                for rank in range(8):
                    with mock.patch.dict(os.environ, SLURM_PROCID=str(rank),
                                         SLURM_STEP_NUM_TASKS="8"):
                        self.assertEqual(launcher.run_rank(work), int(rank == 7))
                    rows = json.loads((work / "ranks" / (str(rank) + ".json")).read_text())["cases"]
                    self.assertEqual([row["case_id"] for row in rows], case_ids[rank::8])
                    self.assertEqual([row["status"] for row in rows],
                        ["FAILED", "NOT_RUN"] if rank == 7 else ["PFC_SOURCE_EXECUTION_COMPLETE"] * 2)
            self.assertEqual(len(invoked), 15)
            self.assertEqual(len({case for case, _ in invoked}), 15)
            self.assertNotIn(case_ids[15], [case for case, _ in invoked])
            self.assertTrue(all(case_ids.index(case) % 8 == rank for case, rank in invoked))


class SourceBatchTests(unittest.TestCase):
    """Exercise real coordinator staging/copyout with simulated rank ledgers only."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        for name in launcher._MODULES:
            path = self.repo / "Libs/population_dynamics" / (name + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# disposable module " + name + "\n")
        wrapper = self.repo / "Output/neural_population_dynamics/operations/pfc_source_execution.sbatch"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("# disposable wrapper\n")
        self.config = self.root / "config.yaml"
        self.config.write_text(json.dumps(dict(cases=[dict(case_id="first"), dict(case_id="second")])))
        self.cases = self.root / "cases.json"
        self.cases.write_text('["second", "first"]')
        self.qualification = self.root / "qualification"
        (self.qualification / "comparison").mkdir(parents=True)
        for name in ("delivery.json", "comparison/qualification.json"):
            (self.qualification / name).write_text("{}")
        self.jobs = 0
        self.environment = dict(NEURONSIM_PROJECT_ROOT=str(self.repo), NEURONSIM_PFC_SOURCE_BACKEND="numpy",
            SLURMD_NODENAME="cpu03", SLURM_NTASKS="2", SLURM_CPUS_PER_TASK="1",
            NEURONSIM_PFC_SOURCE_CASES_JSON=str(self.cases), NEURONSIM_PFC_SOURCE_CONFIG=str(self.config),
            NEURONSIM_PFC_SOURCE_QUALIFICATION=str(self.qualification),
            NEURONSIM_PFC_OUTPUT_BASE=str(self.root / "archives"), NEURONSIM_PFC_SCRATCH=str(self.root / "scratch"))

    def bindings(self):
        files = {"Libs/population_dynamics/" + name + ".py": sha256_file(
                 self.repo / "Libs/population_dynamics" / (name + ".py")) for name in launcher._MODULES}
        files["Libs/scripts/run_pfc_source_cases.py"] = sha256_file(Path(launcher.__file__))
        return {launcher._BINDING_ENV[key]: value for key, value in dict(
            config_sha256=sha256_file(self.config), cases_sha256=sha256_file(self.cases),
            code_sha256=launcher.code_fingerprint(files)).items()}

    def successful_step(self, command, *, cwd, env, omit_case=None, remove_metadata=None):
        work = Path(cwd)
        plan = json.loads((work / "plan.json").read_text())
        for rank in range(plan["tasks"]):
            rows = []
            for case_id in plan["case_ids"][rank::plan["tasks"]]:
                row = dict(case_id=case_id, rank=rank, status="PFC_SOURCE_EXECUTION_COMPLETE",
                           exit_code=0, source_execution_complete=True,
                           config_sha256=sha256_file(work / "source_config.yaml"))
                rows.append(row)
                if case_id == omit_case:
                    continue
                case = work / "cases" / case_id
                case.mkdir(parents=True)
                shutil.copytree(work / "Libs", case / "Libs")
                shutil.copytree(work / "qualification", case / "qualification")
                for name in ("source_config.yaml", "launcher.sbatch", "run_pfc_source_cases.py"):
                    shutil.copy2(work / name, case / name)
                launcher.write_json(case / "execution.json", row)
                source = case / "source_run"
                source.mkdir()
                outputs = {}
                for name in launcher._SOURCE_OUTPUTS:
                    path = source / name
                    path.write_bytes((work / "source_config.yaml").read_bytes() if name == "config.yaml"
                                     else b"disposable metadata-binding fixture")
                    outputs[name] = dict(path=name, sha256=sha256_file(path), bytes=path.stat().st_size)
                launcher.write_json(source / "report.json", dict(row, outputs=outputs,
                                    validations={"software_fixture": {"passed": True}}))
                launcher.write_json(source / "engine_observation.json", dict(engine_pid=100 + rank))
                if remove_metadata is not None and case_id == plan["case_ids"][0]:
                    (case / remove_metadata).unlink()
            launcher.write_json(work / "ranks" / (str(rank) + ".json"), dict(cases=rows, exit_code=0))
        return subprocess.CompletedProcess(command, 0)

    def run_coordinator(self, bindings=None, *, mode="packed", step=None):
        self.jobs += 1
        job = str(200 + self.jobs)
        env = dict(self.environment, SLURM_JOB_ID=job, **(bindings or {}))
        if mode != "packed":
            env.pop("NEURONSIM_PFC_SOURCE_CASES_JSON")
            env["NEURONSIM_PFC_SOURCE_CASE_ID" if mode == "single" else "SLURM_ARRAY_TASK_ID"] = (
                "second" if mode == "single" else "0")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("sys.argv", ["run_pfc_source_cases"]), \
             mock.patch.object(sys, "path", list(sys.path)), \
             mock.patch.object(launcher, "health"), \
             mock.patch.object(launcher.shutil, "disk_usage", return_value=mock.Mock(free=64 * 1024**3)), \
             mock.patch.object(launcher.subprocess, "run", side_effect=step or self.successful_step) as source:
            code = launcher.main()
        batch = self.root / "archives/batches" / ("run_" + job)
        report = json.loads((batch / "delivery.json").read_text())
        return code, report, batch, source

    def test_correct_frozen_bindings_accept_and_archive_selected_bytes(self):
        expected = self.bindings()
        code, report, batch, source = self.run_coordinator(expected)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["case_ids"], ["second", "first"])
        binding = report["input_bindings"]
        self.assertEqual(binding["status"], "VERIFIED")
        self.assertEqual(binding["actual"], binding["expected"])
        self.assertEqual(set(binding["code_files"]), set(launcher._CODE_PATHS))
        self.assertIn("Libs/population_dynamics/pfc_kernel_parameters.py", binding["code_files"])
        self.assertEqual((batch / "cases.json").read_bytes(), self.cases.read_bytes())
        source.assert_called_once()
        self.assertTrue(all(row["archive_status"] == "COMPLETE" for row in report["cases"]))
        self.assertFalse(Path(report["runtime"]).exists())

    def test_selection_uses_frozen_staged_bytes_if_live_inputs_change_after_copy(self):
        expected = self.bindings()
        copy = shutil.copy2
        def changed_live_input(source, destination, *args, **kwargs):
            result = copy(source, destination, *args, **kwargs)
            if Path(source) == self.config:
                self.config.write_text('{"cases": []}')
            elif Path(source) == self.cases:
                self.cases.write_text('["not_registered"]')
            return result
        with mock.patch.object(launcher.shutil, "copy2", side_effect=changed_live_input):
            code, report, _, source = self.run_coordinator(expected)
        self.assertEqual(code, 0)
        self.assertEqual(report["case_ids"], ["second", "first"])
        self.assertEqual(report["input_bindings"]["status"], "VERIFIED")
        source.assert_called_once()

    def test_config_case_list_and_all_runtime_module_drift_reject_before_source_launch(self):
        paths = [self.config, self.cases, *(self.repo / "Libs/population_dynamics" / (name + ".py")
                                           for name in launcher._MODULES)]
        for path in paths:
            with self.subTest(path=path.name):
                expected, original = self.bindings(), path.read_bytes()
                path.write_bytes(original + b"\n ")
                try:
                    code, report, _, source = self.run_coordinator(expected)
                finally:
                    path.write_bytes(original)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(report["input_bindings"]["status"], "FAILED")
                self.assertIn("binding mismatch", report["error"])
                source.assert_not_called()

    def test_staged_launcher_drift_rejects_before_source_launch(self):
        expected, copy = self.bindings(), shutil.copy2
        def changed_launcher(source, destination, *args, **kwargs):
            result = copy(source, destination, *args, **kwargs)
            if Path(source) == Path(launcher.__file__):
                destination = Path(destination)
                destination.write_bytes(destination.read_bytes() + b"\n# fixture drift\n")
            return result
        with mock.patch.object(launcher.shutil, "copy2", side_effect=changed_launcher):
            code, report, _, source = self.run_coordinator(expected)
        self.assertEqual(code, 1)
        self.assertIn("code_sha256", report["error"])
        source.assert_not_called()

    def test_partial_empty_malformed_bindings_and_non_case_list_reject_before_source_launch(self):
        expected = self.bindings()
        names = list(expected)
        invalid = [{names[0]: expected[names[0]]}, {key: value for key, value in expected.items() if key != names[2]}]
        invalid += [dict(expected, **{names[0]: value}) for value in ("", "0" * 63, "g" * 64)]
        for values in invalid:
            with self.subTest(values=values):
                code, report, _, source = self.run_coordinator(values)
                self.assertEqual(code, 1)
                self.assertEqual(report["input_bindings"]["status"], "FAILED")
                self.assertIn("All three frozen input bindings", report["error"])
                source.assert_not_called()
        code, report, _, source = self.run_coordinator(expected, mode="single")
        self.assertEqual(code, 1)
        self.assertIn("explicit case-list", report["error"])
        source.assert_not_called()

    def test_unbound_single_array_and_case_list_keep_their_archive_interfaces(self):
        for mode, selected in (("single", ["second"]), ("array", ["first"]), ("packed", ["second", "first"])):
            with self.subTest(mode=mode):
                code, report, _, source = self.run_coordinator(mode=mode)
                self.assertEqual(code, 0)
                self.assertEqual(report["input_bindings"]["status"], "UNBOUND")
                self.assertEqual(report["case_ids"], selected)
                for row in report["cases"]:
                    expected = "run_" + report["job_id"] + ("_" + row["case_id"] if mode == "packed" else "")
                    self.assertEqual(Path(row["archive"]).name, expected)
                source.assert_called_once()

    def assert_incomplete_archive_retains_sibling(self, code, report):
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAILED")
        failed, sibling = report["cases"]
        self.assertEqual(failed["status"], "PFC_SOURCE_EXECUTION_COMPLETE")
        self.assertEqual(failed["exit_code"], 0)
        self.assertTrue(failed["source_execution_complete"])
        self.assertEqual(failed["archive_status"], "FAILED")
        self.assertEqual(sibling["archive_status"], "COMPLETE")
        self.assertTrue((Path(sibling["archive"]) / "delivery.json").is_file())
        self.assertTrue(Path(report["runtime"]).is_dir())
        if "archive" in failed:
            self.assertFalse((Path(failed["archive"]) / "delivery.json").exists())

    def test_missing_successful_case_directory_fails_batch_and_retains_sibling(self):
        def step(*args, **kwargs):
            return self.successful_step(*args, **kwargs, omit_case="second")
        code, report, _, _ = self.run_coordinator(self.bindings(), step=step)
        self.assert_incomplete_archive_retains_sibling(code, report)
        self.assertIn("Successful case directory missing", report["cases"][0]["archive_error"])

    def test_missing_successful_case_metadata_or_output_fails_batch_and_retains_sibling(self):
        for name in ("execution.json", "source_config.yaml", "source_run/report.json",
                     "source_run/engine_observation.json", "source_run/request.json", "source_run/raw.npz"):
            with self.subTest(name=name):
                def step(*args, **kwargs):
                    return self.successful_step(*args, **kwargs, remove_metadata=name)
                code, report, _, _ = self.run_coordinator(self.bindings(), step=step)
                self.assert_incomplete_archive_retains_sibling(code, report)

    def test_contradictory_successful_source_validations_fail_batch_and_retain_sibling(self):
        for validations in ({"fixture": {"passed": False}}, {"fixture": {"passed": 1}},
                            {"fixture": {"passed": None}}, {"fixture": {}}, {"fixture": True}, {}, None):
            with self.subTest(validations=validations):
                def step(*args, **kwargs):
                    result = self.successful_step(*args, **kwargs)
                    path = Path(kwargs["cwd"]) / "cases/second/source_run/report.json"
                    report = json.loads(path.read_text())
                    report["validations"] = validations
                    launcher.write_json(path, report)
                    return result
                code, report, _, _ = self.run_coordinator(self.bindings(), step=step)
                self.assert_incomplete_archive_retains_sibling(code, report)
                self.assertIn("every declared source validation", report["cases"][0]["archive_error"])

    def test_case_copy_failure_fails_batch_and_retains_sibling(self):
        archive_tree = launcher.archive_tree
        def failed_archive(source, target):
            if source.name == "second":
                raise OSError("deliberate copy failure")
            return archive_tree(source, target)
        with mock.patch.object(launcher, "archive_tree", side_effect=failed_archive):
            code, report, _, _ = self.run_coordinator(self.bindings())
        self.assert_incomplete_archive_retains_sibling(code, report)
        self.assertIn("deliberate copy failure", report["cases"][0]["archive_error"])


    def test_canonical_launcher_is_retained_in_batch_and_case_provenance(self):
        canonical = self.repo / "Libs/scripts/run_pfc_source_cases.sbatch"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("# canonical launcher fixture\n")
        code, report, batch, source = self.run_coordinator()
        self.assertEqual(code, 0)
        source.assert_called_once()
        self.assertEqual((batch / "launcher.sbatch").read_bytes(), canonical.read_bytes())
        for row in report["cases"]:
            self.assertEqual((Path(row["archive"]) / "launcher.sbatch").read_bytes(),
                             canonical.read_bytes())

    def test_legacy_only_staged_tree_keeps_launcher_provenance(self):
        legacy = self.repo / "Output/neural_population_dynamics/operations/pfc_source_execution.sbatch"
        code, _, batch, source = self.run_coordinator()
        self.assertEqual(code, 0)
        source.assert_called_once()
        self.assertEqual((batch / "launcher.sbatch").read_bytes(), legacy.read_bytes())

    def test_missing_launcher_provenance_rejects_before_source_execution(self):
        legacy = self.repo / "Output/neural_population_dynamics/operations/pfc_source_execution.sbatch"
        legacy.unlink()
        code, report, _, source = self.run_coordinator()
        self.assertNotEqual(code, 0)
        source.assert_not_called()
        self.assertIn("pfc_source_execution.sbatch", report["error"])
        self.assertEqual(report["status"], "FAILED")


class ReadoutSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        text = (_ROOT / "Output/neural_population_dynamics/operations/pfc_source_readouts.sbatch").read_text()
        script = text.split('cat > "$work/stage_sources.py" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        old = "/workspace/NeuronSim/Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/streamed_source_execution"
        self.script = script.replace(old, str(self.root / "archives"))
        self.env = dict(os.environ, NRNSIM_PFC_READOUT_MODE="phases")
        for key in ("NRNSIM_PFC_SOURCE_ARCHIVES_JSON", "NRNSIM_PFC_SOURCE_JOBS"):
            self.env.pop(key, None)

    def archive(self, label, case, job="123"):
        root = self.root / "archives" / label
        root.mkdir(parents=True)
        (root / "data.txt").write_text(case)
        delivery = dict(status="ARCHIVED_PFC_SOURCE_EXECUTION", source_execution_complete=True,
                        job_id=job, case_id=case, files=[dict(path="data.txt", bytes=len(case),
                        sha256=sha256_file(root / "data.txt"))])
        (root / "delivery.json").write_text(json.dumps(delivery))
        return root

    def run_stage(self, label, env):
        work = self.root / label
        work.mkdir()
        (work / "stage_sources.py").write_text(self.script)
        result = subprocess.run([sys.executable, "-B", str(work / "stage_sources.py")], env=env,
                                cwd=work, text=True, capture_output=True)
        return work, result

    def test_packed_archives_with_shared_job_id_keep_exact_selection_order(self):
        a = self.archive("run_123_a", "case_a")
        b = self.archive("run_123_b", "case_b")
        path = self.root / "selection.json"
        path.write_text(json.dumps([str(b), str(a)]))
        work, result = self.run_stage("packed", dict(self.env, NRNSIM_PFC_SOURCE_ARCHIVES_JSON=str(path)))
        self.assertEqual(result.returncode, 0, result.stderr)
        origins = json.loads((work / "input_origins.json").read_text())
        self.assertEqual([row["case_id"] for row in origins], ["case_b", "case_a"])
        self.assertEqual([row["job_id"] for row in origins], [123, 123])
        self.assertEqual((work / "selected_source_paths.txt").read_text().splitlines(),
                         [row["staged"] for row in origins])
        self.assertEqual(len({row["staged"] for row in origins}), 2)

    def test_numeric_job_route_preserves_order_and_rejects_duplicate_jobs(self):
        self.archive("run_123", "case_a")
        self.archive("run_124", "case_b", "124")
        work, result = self.run_stage("numeric", dict(self.env, NRNSIM_PFC_SOURCE_JOBS="124,123"))
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads((work / "input_origins.json").read_text())
        self.assertEqual([row["case_id"] for row in rows], ["case_b", "case_a"])
        self.assertEqual([Path(row["staged"]).name for row in rows], ["run_124", "run_123"])
        _, result = self.run_stage("duplicate_jobs", dict(self.env, NRNSIM_PFC_SOURCE_JOBS="123,123"))
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_path_selection_or_changed_payload_rejects_before_analysis(self):
        archive = self.archive("run_123_a", "case_a")
        path = self.root / "selection.json"
        for label, values, extra in (
            ("duplicate", [str(archive), str(archive / ".." / archive.name)], {}),
            ("relative", ["run_123_a"], {}),
            ("empty", [], {}),
            ("both", [str(archive)], dict(NRNSIM_PFC_SOURCE_JOBS="123")),
        ):
            path.write_text(json.dumps(values))
            work, result = self.run_stage(label, dict(self.env, NRNSIM_PFC_SOURCE_ARCHIVES_JSON=str(path), **extra))
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse((work / "selected_source_paths.txt").exists())
        (archive / "data.txt").write_text("case_b")
        path.write_text(json.dumps([str(archive)]))
        work, result = self.run_stage("corrupted", dict(self.env, NRNSIM_PFC_SOURCE_ARCHIVES_JSON=str(path)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash mismatch", result.stderr)

    def test_dataset_restage_uses_durable_paths_and_preserves_legacy_scripts(self):
        origins = [dict(job_id=123, archive="/durable/run_123_b"), dict(job_id=123, archive="/durable/run_123_a")]
        script = self.root / "stage_sources.py"
        script.write_text(self.script)
        env = dict(NRNSIM_PFC_SOURCE_JOBS="999")
        dataset.configure_source_selection(env, self.root, origins)
        self.assertNotIn("NRNSIM_PFC_SOURCE_JOBS", env)
        self.assertEqual(json.loads(Path(env["NRNSIM_PFC_SOURCE_ARCHIVES_JSON"]).read_text()),
                         [row["archive"] for row in origins])
        script.write_text("# historical numeric stage script")
        dataset.configure_source_selection(env, self.root, [dict(job_id=123), dict(job_id=124)])
        self.assertEqual(env, dict(NRNSIM_PFC_SOURCE_JOBS="123,124"))



class CanonicalSourceLauncherTests(unittest.TestCase):
    """Run the shell entry point with an explicit fake interpreter, never a source engine."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo with spaces"
        runner = self.repo / "Libs/scripts/run_pfc_source_cases.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("raise RuntimeError('native source must not execute in launcher tests')\n")
        self.capture = self.root / "capture.json"
        capture_script = self.root / "capture.py"
        capture_script.write_text(
            "import json,os,sys\nfrom pathlib import Path\n"
            "keys=('NEURONSIM_PROJECT_ROOT','PYTHONPATH','NEURONSIM_PFC_SOURCE_BACKEND',"
            "'CUDA_VISIBLE_DEVICES','SLURM_ARRAY_TASK_ID','SLURM_NTASKS',"
            "'NEURONSIM_PFC_SOURCE_CASE_ID','NEURONSIM_PFC_SOURCE_CASES_JSON')\n"
            "Path(os.environ['NEURONSIM_TEST_CAPTURE']).write_text(json.dumps("
            "dict(arguments=sys.argv[1:],environment={k:os.environ.get(k) for k in keys})))\n")
        self.interpreter = self.root / "python with spaces"
        self.interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -B " +
            shlex.quote(str(capture_script)) + ' "$@"\n')
        self.interpreter.chmod(0o755)
        self.environment = dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"),
            NEURONSIM_PFC_PYTHON=str(self.interpreter), NEURONSIM_PROJECT_ROOT=str(self.repo),
            NEURONSIM_TEST_CAPTURE=str(self.capture), SLURM_JOB_ID="123",
            SLURMD_NODENAME="cpu03", SLURM_JOB_PARTITION="gpu", SLURM_NTASKS="1",
            SLURM_CPUS_PER_TASK="1")

    def run_launcher(self, changes=None):
        env = dict(self.environment)
        for key, value in (changes or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(["/bin/bash", str(_ROOT / "Libs/scripts/run_pfc_source_cases.sbatch")],
            cwd=self.root, env=env, text=True, capture_output=True)
        captured = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return result, captured

    def test_default_submission_resources_and_unbounded_time(self):
        directives = {line[len("#SBATCH "):].strip() for line in
            (_ROOT / "Libs/scripts/run_pfc_source_cases.sbatch").read_text().splitlines()
            if line.startswith("#SBATCH ")}
        self.assertTrue({"--partition=gpu", "--nodelist=cpu03", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=1", "--mem=4G", "--gres=none", "--time=0"} <= directives)

    def test_missing_interpreter_rejects_before_invocation(self):
        result, captured = self.run_launcher({"NEURONSIM_PFC_PYTHON": None})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NEURONSIM_PFC_PYTHON", result.stderr)
        self.assertIsNone(captured)

    def test_relative_missing_and_directory_interpreters_reject(self):
        for value in ("python3", str(self.root / "missing"), str(self.root)):
            with self.subTest(interpreter=value):
                result, captured = self.run_launcher({"NEURONSIM_PFC_PYTHON": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("NEURONSIM_PFC_PYTHON", result.stderr)
                self.assertIsNone(captured)

    def test_nonexecutable_interpreter_rejects_before_invocation(self):
        self.interpreter.chmod(0o644)
        result, captured = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("executable file", result.stderr)
        self.assertIsNone(captured)

    def test_explicit_interpreter_and_project_paths_with_spaces(self):
        result, captured = self.run_launcher({"SLURM_SUBMIT_DIR": str(self.root / "invalid")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured["arguments"], ["-B", str(self.repo / "Libs/scripts/run_pfc_source_cases.py")])
        self.assertEqual(captured["environment"]["NEURONSIM_PROJECT_ROOT"], str(self.repo))
        self.assertEqual(captured["environment"]["PYTHONPATH"], str(self.repo / "Libs"))
        self.assertEqual(captured["environment"]["NEURONSIM_PFC_SOURCE_BACKEND"], "numpy")
        self.assertEqual(captured["environment"]["CUDA_VISIBLE_DEVICES"], "")

    def test_repository_submission_directory_supplies_project_root(self):
        result, captured = self.run_launcher({"NEURONSIM_PROJECT_ROOT": None,
                                             "SLURM_SUBMIT_DIR": str(self.repo)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured["environment"]["NEURONSIM_PROJECT_ROOT"], str(self.repo))

    def test_missing_relative_and_nonrepository_roots_reject(self):
        for changes in ({"NEURONSIM_PROJECT_ROOT": None}, {"NEURONSIM_PROJECT_ROOT": "relative"},
                        {"NEURONSIM_PROJECT_ROOT": str(self.root)}):
            with self.subTest(changes=changes):
                result, captured = self.run_launcher(changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(captured)

    def test_single_array_and_packed_environment_interfaces_remain_available(self):
        for changes in ({"NEURONSIM_PFC_SOURCE_CASE_ID": "case"}, {"SLURM_ARRAY_TASK_ID": "2"},
                        {"NEURONSIM_PFC_SOURCE_CASES_JSON": "/tmp/disposable_cases.json", "SLURM_NTASKS": "8"}):
            with self.subTest(changes=changes):
                result, captured = self.run_launcher(changes)
                self.assertEqual(result.returncode, 0, result.stderr)
                for key, value in changes.items():
                    self.assertEqual(captured["environment"][key], value)

    def test_explicit_cuda_allocation_preserves_visible_device(self):
        for node, partition in (("cpu03", "gpu"), ("cpu04", "cpu"), ("cpu05", "cpu"),
                                ("cpu06", "cpu"), ("gpu03", "gpu1")):
            with self.subTest(node=node, partition=partition):
                result, captured = self.run_launcher({"NEURONSIM_PFC_SOURCE_BACKEND": "cuda",
                    "SLURMD_NODENAME": node, "SLURM_JOB_PARTITION": partition, "SLURM_JOB_GPUS": "0,1",
                    "CUDA_VISIBLE_DEVICES": "0,1", "NEURONSIM_PFC_SOURCE_CASE_ID": "case"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(captured["environment"]["NEURONSIM_PFC_SOURCE_BACKEND"], "cuda")
                self.assertEqual(captured["environment"]["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_unknown_backend_rejects_before_interpreter_invocation(self):
        result, captured = self.run_launcher({"NEURONSIM_PFC_SOURCE_BACKEND": "other"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("numpy or cuda", result.stderr)
        self.assertIsNone(captured)


class CanonicalPackingLauncherTests(unittest.TestCase):
    def test_packing_stages_and_invokes_same_canonical_launcher_without_source_execution(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo = root / "repo"
            package = repo / "Libs/population_dynamics"
            package.mkdir(parents=True)
            for module in (*launcher._MODULES, "test_pfc_source_launcher"):
                (package / (module + ".py")).write_text("# disposable module\n")
            scripts = repo / "Libs/scripts"
            scripts.mkdir()
            for filename in ("run_pfc_source_cases.py", "run_pfc_source_cases.sbatch", "validate_pfc_source_packing.py"):
                (scripts / filename).write_bytes((_ROOT / "Libs/scripts" / filename).read_bytes())
            (scripts / "run_pfc_source_dataset.py").write_text("# disposable unused dataset caller\n")
            operations = repo / "Output/neural_population_dynamics/operations"
            operations.mkdir(parents=True)
            (operations / "pfc_source_readouts.sbatch").write_text("# legacy readout fixture\n")
            config = root / "config.yaml"
            config.write_text(json.dumps(dict(excitatory_neurons=1024, inhibitory_neurons=256,
                duration_ms=800, dt_ms=.025, cases=[dict(case_id="case_" + str(i)) for i in range(8)])))
            output = root / "output"
            arguments = ["validate_pfc_source_packing", "--config", str(config),
                "--qualified-stream", str(root / "qualification"), "--native-validation", str(root / "native"),
                "--output", str(output)]
            invoked = []
            def fake_step(command, *, cwd, env, **kwargs):
                work = Path(cwd)
                self.addCleanup(shutil.rmtree, work, ignore_errors=True)
                if Path(command[-1]).name == "run_tests.py":
                    (work / "tests.json").write_text(json.dumps(dict(tests_run=1, failures=0, errors=0, skipped=0, success=True)))
                    return subprocess.CompletedProcess(command, 0)
                invoked.append((command, (work / "Libs/scripts/run_pfc_source_cases.sbatch").read_bytes()))
                raise RuntimeError("FAKE_STOP_BEFORE_NATIVE_SOURCE")
            env = dict(SLURM_JOB_ID="123", SLURMD_NODENAME="cpu03", SLURM_NTASKS="4", SLURM_CPUS_PER_TASK="1")
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("sys.argv", arguments), \
                 mock.patch.object(packing, "__file__", str(scripts / "validate_pfc_source_packing.py")), \
                 mock.patch.object(sys, "path", list(sys.path)), \
                 mock.patch("population_dynamics.resource_probe.probe", return_value=dict(status="PASS", reasons=[])), \
                 mock.patch.object(packing.subprocess, "run", side_effect=fake_step):
                self.assertEqual(packing.main(), 1)
            self.assertEqual(len(invoked), 1)
            command, staged = invoked[0]
            self.assertEqual(command[0], "bash")
            self.assertTrue(command[1].endswith("/Libs/scripts/run_pfc_source_cases.sbatch"))
            expected = (scripts / "run_pfc_source_cases.sbatch").read_bytes()
            self.assertEqual(staged, expected)
            self.assertEqual((output / "Libs/scripts/run_pfc_source_cases.sbatch").read_bytes(), expected)
            self.assertIn("FAKE_STOP_BEFORE_NATIVE_SOURCE", json.loads((output / "report.json").read_text())["error"])


if __name__ == "__main__":
    unittest.main()

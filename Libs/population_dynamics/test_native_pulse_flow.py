"""Exercise native pulse launcher delivery with real comparisons and archive.

Only native execution, episode loading and prerequisite proofs are fixtures.
All nine comparison calls and their saved arrays run through the real launcher.
These tests supply no native or scientific qualification evidence.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import run_nest_gpu_live_control as runner
from .test_native_pulse_comparison import NativeEpisodeFixture, prediction, reference

SEEDS = (41, 42, 43)
CANDIDATES = ("omit", "first_future_pulse", "original_future_pulses")
INITIAL = [1111.6, 1151.6, 1191.6, 1510.0]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class NativePulseFlowTests(unittest.TestCase):
    def _run_flow(self, base, *, wrong_tc=False):
        repo, work, target = base / "repo", base / "work", base / "delivery"
        repo.mkdir()
        source = NativeEpisodeFixture()
        source.pulse_ticks = np.asarray([11126, 11526, 11926, 15110], dtype=np.int64)
        initial_parameters = dict(unit_ids=source.unit_ids, V_m_rel=np.zeros(100000, dtype=np.float32),
                                  Theta_rel=np.full(100000, 15.0, dtype=np.float32),
                                  t_ref=np.full(100000, 2.0, dtype=np.float32))
        case = dict(case_id=runner.COUPLED_CASE, split="train", block=14, template_id="familiar_0",
                    initial_state_seed=83015, background_seed=84015,
                    settings=dict(pulse_control=dict(initial_emission_ms=list(INITIAL)),
                                  duration_ms=6000.0, dt_ms=0.1))
        manifest = dict(resolved_source_network=dict(thalamic_input=True, thalamic_input_type="pulses",
                                                    th_pulse_start=3.0, th_interval=7.0,
                                                    th_delay_pulse_generator=1.0))
        schedules = ([], INITIAL[:1], INITIAL[:3])
        programs = {name: dict(initial_emission_ms=list(INITIAL), decision_time_ms=1000.0,
                               future_emission_ms=list(schedule))
                    for name, schedule in zip(CANDIDATES, schedules)}
        selected = dict(zip(SEEDS, CANDIDATES))
        fields_root = repo / runner.COHORT / "train_fields" / runner.COUPLED_CASE
        _write_json(fields_root / "summary.json", dict(fixture_only=True))
        np.savez_compressed(fields_root / "spatial_fields.npz", field_ids=source.field_ids)
        rows = []
        for name in ("summary.json", "spatial_fields.npz"):
            path = fields_root / name
            rows.append(dict(path=str(path.relative_to(repo / runner.COHORT)),
                             bytes=path.stat().st_size, sha256=runner.digest(path)))
        _write_json(repo / runner.COHORT / "delivery.json", dict(files=rows, fixture_only=True))
        launched = []
        loaded = []

        def stage_coupled(_repo, local, _native_build):
            self.assertEqual(_repo, repo)
            (local / "Libs").mkdir()
            _write_json(local / "inputs/design.json", dict(cases=[case]))
            _write_json(local / "reference/report.json", dict(fixture_only=True))
            np.savez_compressed(local / "reference/initial_parameters.npz", **initial_parameters)
            for name in ("source_rng.so", "libnestgpukernel.so", "nestgpu.py"):
                (local / name).write_bytes(b"native execution fixture\n")
            plan = dict(case_id=runner.COUPLED_CASE, graph_manifest_sha256="a" * 64,
                        native_library_sha256="c" * 64, original_report_sha256=runner.digest(local / "reference/report.json"),
                        settings=deepcopy(case["settings"]), graph_bytes=0)
            return plan, manifest

        def stage_search(_search, local, _plan):
            _write_json(local / "prerequisites/search_delivery.json", dict(fixture_only=True))
            for seed in SEEDS:
                saved = local / "inputs/search/results" / f"seed_{seed}"
                saved.mkdir(parents=True)
                np.savez_compressed(saved / "reference.npz", **reference(source))
                for factor, name in enumerate(CANDIDATES):
                    np.savez_compressed(saved / (name + ".npz"), **prediction(source, factor))
            return {seed: dict(fixture_only=True) for seed in SEEDS}

        def stage_graph(_repo, local, graph_path, _manifest):
            self.assertEqual(_repo, repo)
            # The production path uses /dev/shm; no graph data are written there
            # by this fixture. The path is passed only to mocked native commands.
            self.assertEqual(local, work)

        def run_command(command, **kwargs):
            module = command[command.index("-m") + 1]
            if module == "population_dynamics.resource_probe":
                _write_json(Path(command[command.index("--output") + 1]), dict(status="PASS", fixture_only=True))
            elif module == "population_dynamics.nest_gpu_spatial":
                output = Path(command[command.index("--output") + 1])
                name = output.parent.name
                launched.append(name)
                output.mkdir(parents=True)
                _write_json(output / "report.json", dict(status="EXECUTED", fixture_only=True,
                    native_live_control=dict(future_schedule_replaced=True, initial_tc_schedule_verified=True)))
                shutil.copyfile(work / "reference/initial_parameters.npz", output / "initial_parameters.npz")
            elif module == "population_dynamics.spatial_activity":
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                _write_json(output / "summary.json", dict(fixture_only=True))
                np.savez_compressed(output / "spatial_fields.npz", field_ids=source.field_ids)
            else:
                raise AssertionError("unexpected native flow command: " + module)
            return types.SimpleNamespace(returncode=0)

        def load_episode(episode, _fields, **kwargs):
            loaded.append(str(episode))
            if episode.name == "reference":
                return source
            self.assertEqual(episode.name, "native")
            self.assertTrue(np.array_equal(kwargs["expected_unit_ids"], source.unit_ids))
            self.assertTrue(np.array_equal(kwargs["expected_unit_field_index"], source.unit_field_index))
            actual = deepcopy(source)
            actual.source_identity["native_library_sha256"] = "c" * 64
            name = episode.parent.name
            actual.pulse_ticks = np.asarray([round(value / 0.1) + 10
                                            for value in programs[name]["future_emission_ms"]], dtype=np.int64)
            if wrong_tc and name == "omit":
                actual.pulse_ticks = np.asarray([11126], dtype=np.int64)
            return actual

        program_module = importlib.import_module("population_dynamics.native_pulse_programs")
        compare_module = importlib.import_module("population_dynamics.native_pulse_comparison")
        episode_module = importlib.import_module("population_dynamics.native_renewal_data")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(runner.os.environ, dict(SLURMD_NODENAME="gpu03",
                SLURM_JOB_ID="999999", SLURM_JOB_GPUS="0")))
            stack.enter_context(mock.patch.object(sys, "path", new=list(sys.path)))
            stack.enter_context(mock.patch.object(runner, "stage_coupled_inputs", side_effect=stage_coupled))
            stack.enter_context(mock.patch.object(runner, "require_native_qualification"))
            stack.enter_context(mock.patch.object(runner, "stage_pulse_software", return_value=dict(job_id="fixture-only")))
            stack.enter_context(mock.patch.object(runner, "stage_pulse_search", side_effect=stage_search))
            stack.enter_context(mock.patch.object(program_module, "native_candidate_protocols",
                                                  return_value=(programs, selected)))
            stack.enter_context(mock.patch.object(episode_module, "NativeRenewalEpisode", side_effect=load_episode))
            compare_spy = stack.enter_context(mock.patch.object(compare_module, "compare_candidate",
                                                                wraps=compare_module.compare_candidate))
            stack.enter_context(mock.patch.object(runner, "stage_native_graph", side_effect=stage_graph))
            stack.enter_context(mock.patch.object(runner.subprocess, "run", side_effect=run_command))
            stack.enter_context(mock.patch.object(runner.shutil, "disk_usage",
                return_value=types.SimpleNamespace(free=64 * 1024**3)))
            # Only the fixture graph path is absent; retain real archive and
            # copy_bound operations. Production graph deletion is never executed.
            with mock.patch.object(runner.shutil, "rmtree") as remove_graph:
                code = runner.execute_pulse_search(repo, work, target, base / "build", base / "canary",
                                                    base / "coupled", base / "search", base / "software")
            calls = compare_spy.call_count
        report = json.loads((target / "delivery.json").read_text())
        return code, report, target, launched, loaded, calls

    def test_all_three_native_programs_compare_all_three_seeds_and_archive_actual_arrays(self):
        with tempfile.TemporaryDirectory(prefix="neuronsim-pulse-flow-", dir="/tmp/neuronsim") as temporary:
            code, report, target, launched, loaded, comparisons = self._run_flow(Path(temporary))
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "EXECUTED_DEVELOPMENT")
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual(launched, list(CANDIDATES))
            self.assertEqual(len(loaded), 4)
            self.assertEqual(comparisons, 9)
            self.assertEqual(len(report["candidates"]), 3)
            self.assertEqual(len(report["comparisons"]), 9)
            self.assertEqual(report["native_initial_noise_blocks"], 1)
            for flag in ("training_performed", "scientific_claim_qualified", "independent_intervention_qualified",
                         "success_tolerance_qualified", "native_future_noise_pairing_verified"):
                self.assertIs(report[flag], False)
            for name in CANDIDATES:
                root = target / "candidates" / name
                with np.load(root / "actual_response.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["unit_total_counts"].shape, (100000,))
                    self.assertEqual(int(arrays["unit_total_counts"].sum()), 4)
                    self.assertEqual(arrays["field_rate_hz"].shape, (100, 128))
                    self.assertEqual(arrays["terminal_ready_in_ticks"][0], 20)
                for seed in SEEDS:
                    saved = json.loads((root / f"comparison_seed_{seed}.json").read_text())
                    self.assertEqual(saved["training_seed"], seed)
                    self.assertEqual(saved["candidate"], name)
                    self.assertIs(saved["scientific_claim_qualified"], False)
            self.assertFalse((target / "reference").exists())
            self.assertFalse((target / "reference_fields").exists())
            paths = {row["path"] for row in report["files"]}
            self.assertIn("candidates/omit/actual_response.npz", paths)
            self.assertNotIn("inputs/search/results/seed_41/omit.npz", paths)

    def test_actual_TC_schedule_mismatch_stops_and_archives_first_candidate_failure(self):
        with tempfile.TemporaryDirectory(prefix="neuronsim-pulse-flow-reject-", dir="/tmp/neuronsim") as temporary:
            code, report, target, launched, _loaded, comparisons = self._run_flow(Path(temporary), wrong_tc=True)
            self.assertNotEqual(code, 0)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["exit_code"], code)
            self.assertIn("actual native TC arrivals", report["error"])
            self.assertEqual(launched, ["omit"])
            self.assertEqual(comparisons, 0)
            self.assertEqual(len(report["candidates"]), 1)
            self.assertEqual(report["comparisons"], [])
            self.assertTrue((target / "candidates/omit/native/report.json").is_file())
            self.assertTrue((target / "candidates/omit/native/initial_parameters.npz").is_file())
            self.assertFalse((target / "candidates/omit/actual_response.npz").exists())
            self.assertFalse((target / "candidates/first_future_pulse").exists())
            self.assertFalse((target / "candidates/original_future_pulses").exists())
            self.assertIs(report["scientific_claim_qualified"], False)


if __name__ == "__main__":
    unittest.main()

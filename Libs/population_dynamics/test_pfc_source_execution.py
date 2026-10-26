"""Tiny source execution checks; no native or large-scale qualification claim."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from . import pfc_source_execution as source
from .pfc_native_drive import create_drive
from .provenance import sha256_file


def setting():
    case = dict(case_id="observed_current_test", stimulus_strength_nA=0.5, distractor_strength_nA=0.0)
    config = dict(scope="source_reproduction", expected_source_revision=source._SOURCE_REVISION,
                  synapse_execution="factorized_uniform_projections", excitatory_neurons=8,
                  inhibitory_neurons=2, weight_scaling_factor=256.0, dt_ms=0.05, duration_ms=4.05,
                  stimulus_center_deg=0.0, stimulus_width_deg=90.0, stimulus_start_ms=1.0,
                  stimulus_duration_ms=2.0, stimulus_strength_nA=0.06, random_stream_policy="source_global",
                  source_rng_seed=71, cases=[case])
    return config, case


def observed_result():
    config, case = setting()
    initial, drive = create_drive(config, case)
    result = source.executor.simulate(config, case, initial, drive=drive, observe_applied_current=True)
    return config, result, drive


class AppliedCurrentExecutionTests(unittest.TestCase):
    def test_result_validation_rejects_malformed_current_observations(self):
        config, original, drive = observed_result()
        self.assertTrue(all(row["passed"] is True for row in source.validate_result(config, original, drive).values()))

        def mutate(result, kind):
            observation = result["applied_current_observation"]
            arrays = observation["arrays"]
            if kind == "missing_observation":
                del result["applied_current_observation"]
            elif kind == "missing_array":
                del arrays["current_A"]
            elif kind == "wrong_dtype":
                arrays["current_A"] = arrays["current_A"].astype(np.float32)
            elif kind == "wrong_shape":
                arrays["current_A"] = arrays["current_A"][:, :-1]
            elif kind == "source_identity":
                arrays["source_unit_indices"] = arrays["source_unit_indices"][::-1].copy()
            elif kind == "archive_identity":
                arrays["archive_unit_indices"][0] = 100
            elif kind == "nonfinite":
                arrays["current_A"][1, 0] = np.nan
            elif kind == "negative":
                arrays["current_A"][1, 0] = -1e-9
            elif kind == "units":
                observation["metadata"]["units"] = "nA"
            elif kind == "timing":
                observation["metadata"]["timing"] = "after_rk2"
            elif kind == "gap":
                arrays["segment_stop_ticks"][0] -= 1
            elif kind == "missing_transition":
                for key in ("segment_start_ticks", "segment_stop_ticks", "cue_active", "distractor_active", "current_A"):
                    arrays[key] = arrays[key][[0, 2]].copy()
                arrays["segment_stop_ticks"][0] = arrays["segment_start_ticks"][1]
            elif kind == "active_state":
                arrays["cue_active"][0] = True
            elif kind == "terminal_disagreement":
                arrays["current_A"][-1, 0] = 1e-9
            elif kind == "missing_cost":
                del result["cost"]["applied_current_capture_seconds"]

        for kind in ("missing_observation", "missing_array", "wrong_dtype", "wrong_shape", "source_identity",
                     "archive_identity", "nonfinite", "negative", "units", "timing", "gap", "missing_transition",
                     "active_state", "terminal_disagreement", "missing_cost"):
            with self.subTest(kind=kind):
                result = copy.deepcopy(original)
                mutate(result, kind)
                checks = source.validate_result(config, result, drive)
                self.assertTrue(any(name.startswith("applied_current_") and row["passed"] is False
                                    for name, row in checks.items()), checks)

    def test_execute_persists_current_arrays_metadata_and_manifest_bindings(self):
        config, case = setting()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "execution"
            # Exercise real drive, simulation, validation and serialization; the
            # separately tested native qualification boundary is outside this fixture.
            with mock.patch.object(source, "_qualification", return_value={"scope": "unit-test boundary stub"}):
                report = source.execute(config_path, case["case_id"], output, root / "qualification", "numpy")
            self.assertEqual(report["status"], "PFC_SOURCE_EXECUTION_COMPLETE", report.get("failure"))
            self.assertTrue(report["source_execution_complete"])
            self.assertTrue(all(row["passed"] is True for row in report["validations"].values()))
            self.assertEqual(report["resolved_config"]["stimulus_strength_nA"], 0.06)
            self.assertEqual(report["selected_case"]["stimulus_strength_nA"], 0.5)
            for name in ("applied_current_observation.npz", "applied_current_metadata.json"):
                binding = report["outputs"][name]
                self.assertEqual(binding["path"], name)
                self.assertEqual(binding["sha256"], sha256_file(output / name))
                self.assertEqual(binding["bytes"], (output / name).stat().st_size)
            with np.load(output / "applied_current_observation.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["segment_start_ticks"], [0, 20, 60])
                np.testing.assert_array_equal(arrays["segment_stop_ticks"], [20, 60, 81])
                expected = np.zeros((3, 8))
                expected[1, [7, 0, 1]] = 0.5 * 1e-9
                np.testing.assert_array_equal(arrays["current_A"], expected)
            metadata = json.loads((output / "applied_current_metadata.json").read_text())
            self.assertEqual(metadata["units"], "A")
            self.assertEqual(metadata["population"], "E")
            self.assertEqual(metadata["timing"], "after_source_stimulus_operation_before_rk2")
            unit_metadata = json.loads((output / "unit_metadata.json").read_text())
            self.assertEqual(unit_metadata["applied_current_observation_file"], "applied_current_observation.npz")
            self.assertEqual(unit_metadata["applied_current_metadata_file"], "applied_current_metadata.json")
            capture_seconds = report["costs"]["applied_current_capture_seconds"]
            self.assertGreaterEqual(capture_seconds, 0.0)
            self.assertLessEqual(capture_seconds, report["costs"]["simulator_call_seconds"])
            for key in ("scientific_fidelity_qualified", "working_memory_qualified", "fidelity_at_requested_scale_qualified"):
                self.assertFalse(report[key])
            self.assertEqual(report["production_episodes"], 0)
            persisted = json.loads((output / "report.json").read_text())
            self.assertEqual(persisted["outputs"], report["outputs"])

    def test_cli_rejects_nonfinite_current_with_failed_status_and_nonzero_exit(self):
        config, case = setting()
        original_simulate = source.executor.simulate

        def corrupt_observation(*args, **kwargs):
            result = original_simulate(*args, **kwargs)
            result["applied_current_observation"]["arrays"]["current_A"][1, 0] = np.nan
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = root / "execution"
            arguments = ["pfc_source_execution", "--config", str(config_path), "--case-id", case["case_id"],
                         "--output", str(output), "--qualification", str(root / "qualification"), "--backend", "numpy"]
            with mock.patch.object(source, "_qualification", return_value={"scope": "unit-test boundary stub"}), \
                    mock.patch.object(source.executor, "simulate", side_effect=corrupt_observation), \
                    mock.patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
                status = source.main()
            self.assertEqual(status, 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "PFC_SOURCE_EXECUTION_VALIDATION_FAILED")
            self.assertFalse(report["source_execution_complete"])
            self.assertEqual(report["completed_case_count"], 0)
            self.assertFalse(report["validations"]["applied_current_finite_nonnegative_values"]["passed"])
            with np.load(output / "applied_current_observation.npz", allow_pickle=False) as arrays:
                self.assertTrue(np.isnan(arrays["current_A"][1, 0]))
            self.assertEqual(report["outputs"]["applied_current_observation.npz"]["sha256"],
                             sha256_file(output / "applied_current_observation.npz"))



class QualificationClockTests(unittest.TestCase):
    def qualification(self,root,clocks=None,*,legacy=True,kernel=None,include_kernel=True,helper_sha=None):
        path=root/'comparison/qualification.json';path.parent.mkdir(parents=True,exist_ok=True)
        receipt=dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",
            source_entry_sha256=source.native_drive.SOURCE_ENTRY_SHA256,backend="numpy",
            numpy_version=np.__version__,executor_sha256=sha256_file(Path(source.executor.__file__)),
            drive_sha256=sha256_file(Path(source.native_drive.__file__)),
            input_replay_status="PASS_NATIVE_NUMPY_DRIVE_REPLAY",input_replay_passed_comparisons=24,
            global_cases_passed=2,global_arrays_passed=10)
        if include_kernel:
            receipt['validated_kernel_parameters'] = [source.kernel_parameters.resolve_kernel_parameters(
                {} if kernel is None else kernel)]
            receipt['kernel_parameterization_sha256'] = (helper_sha if helper_sha is not None
                else sha256_file(Path(source.kernel_parameters.__file__)))
        if legacy:
            receipt['global_reference_report_sha256']='99f1f6e8f93499dc0fa6310ade1ecc6cb6364514050542d9e6034d8b69b9814a'
        if clocks is not None:receipt['validated_dt_ms']=clocks
        path.write_text(json.dumps(receipt))
        (root/'delivery.json').write_text(json.dumps(dict(status="ARCHIVED_PFC_STREAM_VALIDATION",
            result_status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",
            files=[dict(path="comparison/qualification.json",bytes=path.stat().st_size,sha256=sha256_file(path))])))
        return root

    def test_legacy_receipt_only_admits_its_bound_native_clock(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.qualification(root)
            self.assertEqual(source._qualification(root,'numpy',.05)['qualification']['global_cases_passed'],2)
            with self.assertRaisesRegex(ValueError,'CLOCK_NOT_QUALIFIED'):source._qualification(root,'numpy',.025)
            self.qualification(root,legacy=False)
            with self.assertRaisesRegex(ValueError,'CLOCK_NOT_QUALIFIED'):source._qualification(root,'numpy',.05)

    def test_explicit_single_clock_scope_is_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.qualification(root,[.025],legacy=False)
            self.assertEqual(source._qualification(root,'numpy',.025)['qualification']['validated_dt_ms'],[.025])
            for declared in ([],[True],[0.],[-.025],[.025,.05],[float('nan')]):
                with self.subTest(declared=declared):
                    self.qualification(root,declared,legacy=False)
                    with self.assertRaisesRegex(ValueError,'CLOCK_NOT_QUALIFIED'):
                        source._qualification(root,'numpy',.025)

    def test_cli_rejects_unqualified_refinement_before_random_initialization(self):
        config,case=setting();config['dt_ms']=.025
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);config_path=root/'config.yaml';config_path.write_text(yaml.safe_dump(config))
            qualification=self.qualification(root/'qualification')
            output=root/'execution'
            arguments=['pfc_source_execution','--config',str(config_path),'--case-id',case['case_id'],
                       '--output',str(output),'--qualification',str(qualification),'--backend','numpy']
            with mock.patch.object(source.native_drive,'create_drive') as initialize, \
                 mock.patch('sys.argv',arguments),redirect_stdout(io.StringIO()):
                status=source.main()
            self.assertEqual(status,1);initialize.assert_not_called()
            report=json.loads((output/'report.json').read_text())
            self.assertFalse(report['source_execution_complete'])
            self.assertIn('CLOCK_NOT_QUALIFIED',str(report['failure']))
            self.assertFalse((output/'initial_voltage.npy').exists())



    def test_missing_wrong_or_changed_kernel_coverage_precedes_initialization(self):
        for name, kernel, included, helper in (
                ('uncovered', dict(sigma_weight_profile=30.), True, None),
                ('missing', {}, False, None),
                ('changed_helper', {}, True, '0'*64)):
            config, case = setting()
            config.update(kernel)
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); path = root/'config.yaml'; path.write_text(yaml.safe_dump(config))
                qualification = self.qualification(root/'qualification', [.05], legacy=False,
                                                  include_kernel=included, helper_sha=helper)
                with mock.patch.object(source.native_drive, 'create_drive') as initialize:
                    report = source.execute(path, case['case_id'], root/'run', qualification, 'numpy')
                initialize.assert_not_called()
                self.assertEqual(report['status'], 'PFC_SOURCE_EXECUTION_FAILED')
                self.assertIn('KERNEL_', report['failure']['message'])
                self.assertFalse((root/'run/initial_voltage.npy').exists())

    def test_explicit_nondefault_execution_persists_tuple_and_weighted_identity(self):
        config, case = setting()
        config.update(sigma_weight_profile=30., Jpos_excit2excit=1.7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root/'config.yaml'; path.write_text(yaml.safe_dump(config))
            qualification = self.qualification(root/'qualification', [.05], legacy=False, kernel=config)
            report = source.execute(path, case['case_id'], root/'run', qualification, 'numpy')
            self.assertEqual(report['status'], 'PFC_SOURCE_EXECUTION_COMPLETE', report.get('failure'))
            kernel = source.kernel_parameters.resolve_kernel_parameters(config)
            provenance = source.kernel_parameters.circuit_provenance(config, source.native_drive.SOURCE_ENTRY_SHA256)
            self.assertEqual(report['resolved_kernel_parameters'], kernel)
            self.assertEqual(report['circuit_provenance'], provenance)
            request = json.loads((root/'run/request.json').read_text())
            metadata = json.loads((root/'run/unit_metadata.json').read_text())
            self.assertEqual(request['resolved_kernel_parameters'], kernel)
            self.assertEqual(metadata['coupling']['resolved_kernel_parameters'], kernel)
            self.assertEqual(metadata['coupling']['circuit_provenance'], provenance)
            self.assertEqual(report['code']['kernel_parameterization_sha256'],
                             sha256_file(Path(source.kernel_parameters.__file__)))


if __name__ == "__main__":
    unittest.main()

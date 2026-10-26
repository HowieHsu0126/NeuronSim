"""Exercise the real intake CLI with disposable metadata and no simulations."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import pfc_native_correspondence_intake as intake
from .provenance import canonical_json, sha256_file


class NativeCorrespondenceIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pfc-intake-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.family = [f"pfc_{block}_{condition}" for block in (94001, 94002, 94003)
                       for condition in ("cue", "omission", "strong_cue")]
        self.reused_id = "pfc_94001_strong_cue"
        self.selected = [name for name in self.family if name != self.reused_id]
        self.native_root = self.root / "native"
        self.preparation = self.root / "preparation"
        self.request_path = self.root / "request.json"
        self.execution_path = self.root / "run_123/execution.json"
        self.reused_path = self.root / "reused.json"
        self.output = self.root / "intake.json"
        self.comparison_path = self.root / "comparison.json"
        self.old_comparison_path = self.root / "old_comparison.json"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.set_scheduler("COMPLETED", "0:0")
        self.flags = dict.fromkeys(intake.FALSE_FLAGS, False) | dict(production_episodes=0)
        self.write(self.root / "canonical.yaml", {"fixture": True})
        self.write(self.preparation / "native_config.yaml", {"fixture": True})
        helper = self.native_root / "Libs/population_dynamics/pfc_factorized_source.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("# disposable helper binding\n")
        self.design = dict(status="REGISTERED_BEFORE_NATIVE_REFERENCE_EXECUTION",
            scope="native_reference_clock_correspondence", native_config_file="native_config.yaml",
            native_config_sha256=sha256_file(self.preparation / "native_config.yaml"),
            source_config_sha256=sha256_file(self.root / "canonical.yaml"),
            reference_execution_implementation_sha256=sha256_file(helper), source_revision="fixture-source",
            source_entry_sha256="a" * 64, reference_synapse_execution="factorized_uniform_projections",
            reference_runtime=dict(brian2_version="2.5.4", numpy_version="1.26.4", scipy_version="1.12.0"),
            excitatory_neurons=80000, inhibitory_neurons=20000, dt_ms=0.0125, duration_ms=5000,
            relative_tolerance=0.0, terminal_voltage_mV_max_absolute_error=1e-10,
            case_ids=self.selected, cases=8, canonical_case_ids=self.family, canonical_registered_cases=9,
            preexisting_source_bindings=[dict(case_id=name, delivery_sha256="b" * 64,
                                             report_sha256="c" * 64) for name in self.selected], **self.flags)
        self.write(self.preparation / "design.json", self.design)
        self.runtime = dict(python="3.10.14 fixture", brian2="2.5.4", numpy="1.26.4", scipy="1.12.0")
        self.native_cases = []
        for name in self.selected:
            raw = self.native_root / "native" / (name + ".npz")
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(b"engineering placeholder; never opened by intake")
            self.native_cases.append(dict(case_id=name, core_units=100000, source_rng_seed=int(name.split("_")[1]),
                raw_file=raw.name, raw_sha256=sha256_file(raw), native_events=2, elapsed_seconds=1.))
        native = dict(status="NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS", cases=self.native_cases,
            source_revision=self.design["source_revision"], source_entry_sha256=self.design["source_entry_sha256"],
            execution_implementation_sha256=sha256_file(helper), synapse_execution=self.design["reference_synapse_execution"],
            resolved_config={key: self.design[key] for key in
                ("excitatory_neurons", "inhibitory_neurons", "dt_ms", "duration_ms")},
            scope="source_reproduction", full_scale_qualified=False,
            scientific_fidelity_qualified=False)
        self.write(self.native_root / "native/report.json", native)
        self.write(self.native_root / "native_config.yaml", {"fixture": True})
        native_execution = dict(status="NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE", job_id="123",
            native_exit_code=0, native_report_sha256=sha256_file(self.native_root / "native/report.json"),
            design_sha256=sha256_file(self.preparation / "design.json"),
            native_config_sha256=self.design["native_config_sha256"],
            source_config_sha256=self.design["source_config_sha256"], runtime_versions=self.runtime, **self.flags)
        self.write(self.native_root / "execution.json", native_execution)
        paths = [helper, self.native_root / "native/report.json", self.native_root / "native_config.yaml",
                 self.native_root / "execution.json"] + [self.native_root / "native" / c["raw_file"] for c in self.native_cases]
        delivery = dict(native_execution, exit_code=0, archive_status="COMPLETE", completed_cases=8,
            files=[dict(path=str(p.relative_to(self.native_root)), **self.binding(p)) for p in paths])
        self.write(self.native_root / "delivery.json", delivery)
        self.request = dict(scientific_execution_authorized=True, production_admission_amendment_authorized=False,
            scientific_protocol_changed=False, source_cases_regenerated=False, production_episodes=0,
            cases=self.selected, reused_completed_case=self.reused_id, reused_completed_job_id=122,
            native_command=["fixture", "--output", str(self.native_root), "--preparation", str(self.preparation),
                "--canonical-config", str(self.root / "canonical.yaml")],
            comparison_output=str(self.comparison_path))
        self.write(self.request_path, self.request)
        tolerance = dict(other_arrays="exact", relative=0.0, terminal_voltage_mV_absolute=1e-10)
        self.comparison = dict(status="PASS_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE",
            canonical_case_ids=self.family, tolerances=tolerance, cases=[self.case(name) for name in self.selected],
            common_execution_binding=dict(fixture="same source execution"),
            source_bindings=self.design["preexisting_source_bindings"], failed_case_ids=[], array_values_compared=80,
            source_revision=self.design["source_revision"], source_entry_sha256=self.design["source_entry_sha256"],
            design_sha256=native_execution["design_sha256"], native_config_sha256=self.design["native_config_sha256"],
            config_sha256=self.design["source_config_sha256"],
            reference_bindings=[dict(delivery_sha256=sha256_file(self.native_root / "delivery.json"),
                report_sha256=native_execution["native_report_sha256"],
                execution_receipt_sha256=sha256_file(self.native_root / "execution.json"),
                helper_sha256=sha256_file(helper))], **self.flags)
        self.write(self.comparison_path, self.comparison)
        old = dict(self.comparison, cases=[self.case(self.reused_id)])
        self.write(self.old_comparison_path, old)
        self.reused = dict(status="PASS_TERMINAL_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE_INTAKE",
            job_id=122, original_case=self.reused_id, source_qualification_granted=False,
            source_history_population_uncertainty_established=False, core_units=100000,
            duration_ms=5000, dt_ms=0.0125, parameter_circuits=1,
            comparison_report=str(self.old_comparison_path),
            metadata_verified={str(self.old_comparison_path): self.binding(self.old_comparison_path)}, **self.flags)
        self.write(self.reused_path, self.reused)
        self.execution = dict(status="COMPLETE_AUTHORIZED_NATIVE_AND_COMPARISON", job_id="123",
            request=self.request, node="cpu03", scientific_protocol_changed=False,
            native_exit_code=0, comparison_exit_code=0, exit_code=0,
            phases=[dict(name=name, exit_code=0) for name in ("native_preflight", "native_validation", "source_comparison")],
            native_delivery_sha256=sha256_file(self.native_root / "delivery.json"),
            comparison_report=str(self.comparison_path), comparison_status=self.comparison["status"], **self.flags)
        self.write(self.execution_path, self.execution)

    def set_scheduler(self, state, code):
        program = self.bin / "sacct"
        program.write_text("#!/bin/sh\nprintf '%s\\n' '123|" + state + "|" + code + "|10|cpu03|'\n")
        program.chmod(0o755)

    def binding(self, p):
        return dict(bytes=p.stat().st_size, sha256=sha256_file(p))

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical_json(value))

    def case(self, name):
        arrays = [dict(array=key, passed=True, unequal_values=0, values_compared=2,
                       atol=1e-10 if key == "terminal_voltage_mV" else 0.0, rtol=0.0)
                  for key in intake.ARRAY_NAMES]
        return dict(case_id=name, source_rng_seed=int(name.split("_")[1]), native_events=2, candidate_events=2,
                    native_seconds=1., candidate_seconds=2., arrays=arrays, values_compared=10,
                    failed_arrays=[], status="PASS")

    def cli(self):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        return subprocess.run([sys.executable, "-B", "-m", intake.__name__,
            "--project-root", str(self.root), "--request", str(self.request_path), "--job-id", "123",
            "--reused-acceptance", str(self.reused_path), "--output", str(self.output)],
            env=env, capture_output=True, text=True)

    def assert_rejected(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def rebind_native_report(self, changes):
        """Update the complete fixture binding chain so eligibility is tested."""
        native_path = self.native_root / "native/report.json"
        native = json.loads(native_path.read_text())
        for key, value in changes.items():
            if value is None:
                native.pop(key, None)
            else:
                native[key] = value
        self.write(native_path, native)
        execution_path = self.native_root / "execution.json"
        execution = json.loads(execution_path.read_text())
        execution["native_report_sha256"] = sha256_file(native_path)
        self.write(execution_path, execution)
        delivery_path = self.native_root / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        delivery["native_report_sha256"] = execution["native_report_sha256"]
        for entry in delivery["files"]:
            entry.update(self.binding(self.native_root / entry["path"]))
        self.write(delivery_path, delivery)
        self.execution["native_delivery_sha256"] = sha256_file(delivery_path)
        self.write(self.execution_path, self.execution)
        self.comparison["reference_bindings"][0].update(
            delivery_sha256=sha256_file(delivery_path),
            report_sha256=sha256_file(native_path),
            execution_receipt_sha256=sha256_file(execution_path))
        self.write(self.comparison_path, self.comparison)

    def test_native_report_explicit_zero_production_accepted(self):
        self.rebind_native_report({"production_episodes": 0})
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_report_eligibility_changes_rejected(self):
        for key, value in (("production_episodes", 1), ("full_scale_qualified", True),
                           ("scientific_fidelity_qualified", True),
                           ("full_scale_qualified", None),
                           ("scientific_fidelity_qualified", None),
                           ("scope", "production")):
            with self.subTest(key=key, value=value):
                self.rebind_native_report({"production_episodes": None,
                    "full_scale_qualified": False, "scientific_fidelity_qualified": False,
                    "scope": "source_reproduction", key: value})
                self.assert_rejected()

    def test_wrapper_missing_production_counter_rejected(self):
        self.execution.pop("production_episodes")
        self.write(self.execution_path, self.execution)
        self.assert_rejected()

    def test_complete_cli_and_idempotent_write(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["canonical_case_ids"], self.family)
        self.assertEqual(report["compared_cases"], 9)
        self.assertEqual(report["array_values_compared"], 90)
        self.assertEqual(report["production_episodes"], 0)
        self.assertFalse(report["scientific_fidelity_qualified"])
        before = self.output.read_bytes()
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(self.output.read_bytes(), before)

    def test_running_job_rejected(self):
        self.set_scheduler("RUNNING", "0:0")
        self.assert_rejected()

    def test_corrupt_delivery_rejected(self):
        p = self.native_root / "delivery.json"
        p.write_text(p.read_text() + " ")
        self.assert_rejected()

    def test_request_mismatch_rejected(self):
        self.execution["request"] = dict(self.request, scientific_execution_authorized=False)
        self.write(self.execution_path, self.execution)
        self.assert_rejected()

    def test_missing_selected_case_rejected(self):
        self.comparison["cases"].pop()
        self.write(self.comparison_path, self.comparison)
        self.assert_rejected()

    def test_duplicate_canonical_case_rejected(self):
        self.comparison["cases"][-1] = copy.deepcopy(self.comparison["cases"][0])
        self.write(self.comparison_path, self.comparison)
        self.assert_rejected()

    def test_tolerance_change_rejected(self):
        self.comparison["tolerances"]["terminal_voltage_mV_absolute"] = 1e-3
        self.write(self.comparison_path, self.comparison)
        self.assert_rejected()

    def test_missing_raw_payload_rejected_without_opening_it(self):
        (self.native_root / "native" / self.native_cases[0]["raw_file"]).unlink()
        self.assert_rejected()

    def test_numerical_failure_retains_all_cases(self):
        case = self.comparison["cases"][0]
        case["arrays"][0]["passed"] = False
        case["failed_arrays"] = [intake.ARRAY_NAMES[0]]
        case["status"] = "FAILED_NUMERICAL_EQUIVALENCE"
        self.comparison["failed_case_ids"] = [case["case_id"]]
        self.comparison["status"] = "FAILED_NATIVE_REFERENCE_CLOCK_CORRESPONDENCE"
        self.write(self.comparison_path, self.comparison)
        self.execution.update(status="COMPLETE_COMPARISON_WITH_NUMERICAL_FAILURE", exit_code=1,
            comparison_exit_code=1, comparison_status=self.comparison["status"])
        self.execution["phases"][-1]["exit_code"] = 1
        self.write(self.execution_path, self.execution)
        self.set_scheduler("FAILED", "1:0")
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["compared_cases"], 9)
        self.assertEqual(report["failed_case_ids"], [case["case_id"]])
        self.assertFalse(report["source_qualification_granted"])


if __name__ == "__main__":
    unittest.main()

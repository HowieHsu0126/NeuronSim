"""Receipt-only fixtures for legacy and Merlin reference-control identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

import v1_reference_preparation_merlin as preparation
from v1_construction_intake_merlin import source_identity


class CohortFixture:
    def identity(self, circuit, block, condition):
        return dict(circuit=circuit, condition=condition)

    def verify_declaration(self, config, identity):
        pass

    def v1_visual_config(self, identity):
        return dict(condition_index=identity["condition"])

    def require_current(self):
        pass


class ReferencePreparationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q = self.root / "qualification"
        self.intake_output = self.root / "intake"
        self.execution = dict(backend="merlin", run_id="source-run", trial_id="source-trial",
                              hostname="source-host", allocated_cpus=4, allocated_gpus=0)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def construction(self, circuit, merlin):
        base = self.q / f"circuit_{circuit:03d}"
        base.mkdir(parents=True)
        config = base / "circuit.yaml"
        config.write_text(yaml.safe_dump(dict(builder_seed=400000000 + circuit)))
        config_sha = preparation.sha256_file(config)
        parent_source, input_source = self.root / "parent", self.root / "source"
        parent_job = str(10000 + circuit)
        input_job = "merlin:source-run:source-trial" if merlin else str(20000 + circuit)
        parent = dict(status="COMPLETE_SOURCE_CONSTRUCTION", exit_code=0, array_task_id=circuit,
                      circuit=dict(config_sha256=config_sha), job_id=parent_job, archive=str(parent_source))
        inputs = dict(status="COMPLETE_SOURCE_CONSTRUCTION", exit_code=0, circuit=dict(config_sha256=config_sha),
                      job_id=input_job, parent_job_id=parent_job, archive=str(input_source))
        if merlin:
            inputs.update(circuit_index=circuit, job_id_namespace="merlin_run_trial", execution_context=self.execution)
        else:
            inputs["array_task_id"] = circuit
        self.write(parent_source / "delivery.json", dict(job_id=parent_job))
        self.write(input_source / "delivery.json", {key: value for key, value in inputs.items()
                   if key in ("job_id", "job_id_namespace", "execution_context")})
        parent["delivery_sha256"] = preparation.sha256_file(parent_source / "delivery.json")
        inputs["delivery_sha256"] = preparation.sha256_file(input_source / "delivery.json")
        records = self.q / "construction_expansion_20261017" / base.name
        self.write(records / "recurrent/report.json", parent)
        self.write(base / "recurrent_acceptance.json", dict(status="PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY",
                   source_job_id=int(parent_job), source=str(parent_source),
                   consumer_binding=dict(delivery_sha256=parent["delivery_sha256"]),
                   source_simulations=0, production_episodes=0))
        record_path, acceptance_path = preparation.input_paths(self.q, circuit, self.intake_output if merlin else None)
        self.write(record_path, inputs)
        acceptance = dict(status="PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY", **source_identity(inputs),
                          recurrent_job_id=int(parent_job), source=str(input_source), config_sha256=config_sha,
                          delivery_sha256=inputs["delivery_sha256"], source_simulations=0, production_episodes=0)
        if merlin:
            acceptance["circuit_index"] = circuit
        self.write(acceptance_path, acceptance)
        return acceptance_path

    def control(self):
        detector = self.root / "detector.py"
        detector.write_text("# preserved detector fixture\n")
        rate, spike = self.root / "rate", self.root / "spike"
        for directory, relative, status in ((rate, "probe/probe.json", "fixture"),
                (spike, "visual_inputs/report.json", "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY")):
            self.write(directory / relative, dict(config={}))
            self.write(directory / "delivery.json", dict(exit_code=0, status=status,
                       files=[dict(path=relative, **preparation.file_identity(directory / relative))]))
        rows = [dict(condition_index=index, rate_code_reference=str(rate), spike_canary=str(spike),
                     rate=dict(delivery_sha256=preparation.sha256_file(rate / "delivery.json")),
                     spike=dict(delivery_sha256=preparation.sha256_file(spike / "delivery.json"))) for index in range(10)]
        return dict(records=rows, native_python="preserved-source-python", source_field="unchanged"), detector

    def prepare(self, circuit, merlin):
        source, detector = self.control()
        with mock.patch.object(preparation, "require_same_recipe") as recipe_check:
            artifacts, proof = preparation.prepare_circuit(self.q, self.root / "prepared", source, detector,
                dict(node="cpu04", partition="cpu"), 16, circuit, CohortFixture(), self.execution,
                self.intake_output if merlin else None)
        self.assertEqual(recipe_check.call_count, 20)
        self.assertEqual(len(artifacts), 12)
        self.assertEqual(proof["prepared_conditions"], 10)
        control = preparation.read_json(Path(proof["prepared_control"]))
        self.assertEqual([row["condition_index"] for row in control["records"]], list(range(10)))
        self.assertEqual(control["native_python"], "preserved-source-python")
        self.assertEqual(control["gpu"], dict(node="cpu04", partition="cpu"))
        self.assertEqual(control["gpu_memory_gib"], 16)
        return control

    def test_merlin_candidate_prepares_ten_conditions_with_namespace(self):
        self.construction(98, True)
        control = self.prepare(98, True)
        self.assertEqual(control["inputs"]["job_id"], "merlin:source-run:source-trial")
        self.assertEqual(control["inputs"]["job_id_namespace"], "merlin_run_trial")
        self.assertEqual(control["inputs"]["execution_context"], self.execution)
        self.assertIs(type(control["parent"]["job_id"]), int)
        self.assertFalse((self.q / "circuit_098/input_acceptance.json").exists())

    def test_legacy_controls_and_default_circuits_keep_original_types(self):
        self.construction(94, False)
        control = self.prepare(94, False)
        self.assertIs(type(control["inputs"]["job_id"]), int)
        self.assertEqual(set(control["inputs"]), {"job_id", "expected_archive", "parent_job_id"})
        self.assertEqual(preparation.selected_circuits(argparse.Namespace()), (94, 95, 96, 97))

    def test_merlin_acceptance_cannot_drop_the_source_namespace(self):
        path = self.construction(98, True)
        value = preparation.read_json(path)
        del value["source_job_id_namespace"]
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "EXISTING_INPUT_ACCEPTANCE_REQUIRED"):
            preparation.accepted_construction(self.q, 98, CohortFixture(), self.intake_output)


if __name__ == "__main__":
    unittest.main()

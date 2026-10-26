"""Bounded metadata fixtures for the nine-construction Merlin intake.

These fixtures exercise receipt authentication. They are not simulated neural
networks or evidence of scientific source qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

import v1_construction_intake_merlin as intake


class CohortFixture:
    def __init__(self, stale=False):
        self.stale = stale
        self.freshness_checks = 0

    def identity(self, circuit, block, condition):
        return dict(connectivity_seed=100 + circuit)

    def verify_declaration(self, config, identity):
        if config["builder_seed"] != identity["connectivity_seed"]:
            raise ValueError("FIXTURE_DECLARATION_DIFFERS")

    def require_current(self):
        self.freshness_checks += 1
        if self.stale:
            raise ValueError("FIXTURE_COHORT_CHANGED")


class ConstructionIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q, self.records, self.sources = (self.root / name for name in ("qualification", "records", "sources"))
        self.q.mkdir()
        self.records.mkdir()
        self.sources.mkdir()
        self.execution = dict(backend="merlin", run_id="fixture-run", trial_id="fixture-trial",
                              hostname="fixture-merlin-host", allocated_cpus=4, allocated_gpus=0)
        self.cohort = CohortFixture()
        rows = [dict(stage=stage, circuit=circuit, raw_job_id=str(job),
                     array_member=intake.ARRAYS[stage] + "_" + str(circuit), state="COMPLETED",
                     exit_code="0:0", elapsed_seconds="10") for stage, circuit, job in intake.TASKS]
        self.accounting = dict(returncode=0, expected_rows=9, observed_utc="fixture-observation",
            matched_rows=rows, stdout="\n".join("|".join((row["raw_job_id"], row["array_member"],
                row["state"], row["exit_code"], row["elapsed_seconds"])) for row in rows))
        cohort_path = self.q / "identities.json"
        accounting_path = self.root / "accounting.json"
        self.write_json(cohort_path, {"fixture": True})
        self.write_json(accounting_path, self.accounting)
        self.context = intake.BoundInputs(self.q, self.records, self.sources,
            cohort_path, intake.sha256_file(cohort_path), accounting_path,
            intake.sha256_file(accounting_path), self.accounting)
        self.shape = self.root / "role.json"
        self.write_json(self.shape, dict(name="head", cpu=4, gpu=0, memory_mb=16384))

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def args(self, output=None):
        return argparse.Namespace(qualification_root=self.q, construction_root=self.records,
            source_root=self.sources, cohort_manifest=self.context.cohort_manifest,
            cohort_manifest_sha256=self.context.cohort_manifest_sha256,
            construction_manifest=self.context.construction_manifest,
            construction_manifest_sha256=self.context.construction_manifest_sha256,
            resource_shape=self.shape, output=output or self.root / "result")

    def config(self, circuit):
        path = self.q / f"circuit_{circuit:03d}" / "circuit.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = dict(builder_seed=100 + circuit)
        path.write_text(yaml.safe_dump(value))
        return value, intake.sha256_file(path)

    def payload(self, root, relative, data=b"bounded fixture payload"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=relative, **intake.file_identity(path))

    def recurrent(self, circuit=95, connected=True, training_seeds=None):
        source_job = 628375 if circuit == 94 else next(job for stage, index, job in intake.TASKS
                                                       if stage == "recurrent" and index == circuit)
        source = self.sources / f"recurrent_{circuit}"
        config, config_sha = self.config(circuit)
        validation = dict(nodes=230924, connectivity=dict(nodes=230924, largest_strong_component=230924,
            strong_components=1 if connected else 2, source_edge_rows=5), all_edge_rows_verified=5,
            edge_rows=5, endpoints_and_nsyns_preserved=True, full_scale_gate_applied=True,
            node_assignment_membership_verified=True, source_and_target_predicates_verified=True)
        report = dict(status="ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE", exit_code=0, mode="full",
            original_config=config, original_config_sha256=config_sha,
            training_seeds_used=[] if training_seeds is None else training_seeds,
            rng=dict(builder_seed=100 + circuit, glif_assignment_seed=1000100 + circuit),
            validation=validation, elapsed_seconds=1)
        for label in ("primary", "replay"):
            self.write_json(source / label / "report.json", report)
        comparison = dict(status="ORIGINAL_RECURRENT_REPLAY_EQUAL", exit_code=0,
            all_native_datasets_and_tables_equal=True, files=[dict(equal=True) for _ in range(8)],
            input_report_sha256=[intake.sha256_file(source / label / "report.json")
                                 for label in ("primary", "replay")])
        self.write_json(source / "comparison/report.json", comparison)
        rows = [dict(path=name, **intake.file_identity(source / name)) for name in
                ("primary/report.json", "replay/report.json", "comparison/report.json")]
        rows.extend(self.payload(source, "primary/" + name) for name in intake.REFERENCE_FILES)
        # This intentionally unstaged edge archive must never be read by intake.
        rows.append(dict(path="primary/biophysical/v1_edges.h5", bytes=5720000000, sha256="0" * 64))
        delivery = dict(exit_code=0, mode="full", job_id=str(source_job),
            status="PASS_SOURCE_RECURRENT_FIXTURE", child_peak_rss_bytes=1, files=rows)
        self.write_json(source / "delivery.json", delivery)
        record = dict(status="COMPLETE_SOURCE_CONSTRUCTION", exit_code=0, array_job_id="624318",
            array_task_id=circuit, job_id=str(source_job), archive=str(source),
            circuit=dict(config_sha256=config_sha), delivery_sha256=intake.sha256_file(source / "delivery.json"))
        self.write_json(self.context.record_path(circuit, "recurrent"), record)
        parent = dict(status="PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY", source_job_id=source_job,
                      consumer_binding=dict(delivery_sha256=record["delivery_sha256"]))
        if circuit == 94:
            self.write_json(self.context.circuit(94) / "recurrent_acceptance.json", parent)
        return source, parent

    def inputs(self, circuit=95, comparison_complete=True, merlin=False):
        parent_source, parent = self.recurrent(circuit)
        source_execution = dict(self.execution, run_id="source-run", trial_id="source-trial")
        source_job = ("merlin:source-run:source-trial" if merlin else
                      next(job for stage, index, job in intake.TASKS if stage == "inputs" and index == circuit))
        source = self.sources / f"inputs_{circuit}"
        config, config_sha = self.config(circuit)
        parent_sha = intake.sha256_file(parent_source / "delivery.json")
        report = dict(status="ORIGINAL_CIRCUIT_INPUTS_COMPLETE", exit_code=0,
            original_config=config, original_config_sha256=config_sha, mode="full",
            stored_v1_files_preserved=True, imported_v1=dict(nodes=230924),
            reference=dict(delivery_sha256=parent_sha, files={"fixture": "same"}),
            rng=dict(lgn_selection_python_seed=100 + circuit), training_seeds_used=[], elapsed_seconds=1,
            validation={population: dict(source_units=units, all_edge_rows_verified=3, edge_rows=3,
                identities_nsyns_and_target_predicates_verified=True) for population, units in (("lgn", 17400), ("bkg", 1))})
        if merlin:
            report["execution_context"] = source_execution
        for label in ("primary", "replay"):
            self.write_json(source / label / "report.json", report)
        expected = [dict(path=name, equal=True) for name in (*intake.NATIVE_FILES, *intake.REFERENCE_FILES)]
        if not comparison_complete:
            expected[-1] = deepcopy(expected[0])
        self.write_json(source / "comparison/report.json", dict(status="ORIGINAL_CIRCUIT_INPUT_REPLAY_EQUAL",
            exit_code=0, all_native_files_equal=True, bound_reference_and_rng_states_equal=True,
            binding_differences=[], files=expected))
        rows = [dict(path=name, **intake.file_identity(source / name)) for name in
                ("primary/report.json", "replay/report.json", "comparison/report.json")]
        rows.extend(self.payload(source, "primary/" + name) for name in (*intake.NATIVE_FILES, *intake.REFERENCE_FILES))
        rows.extend(self.payload(source, f"extra/payload_{index:03d}.bin") for index in range(174 - len(rows)))
        self.write_json(source / "delivery.json", dict(exit_code=0, status="PASS_SOURCE_INPUT_BUILD_REPEATABILITY",
            job_id=str(source_job), mode="full", recurrent_reference=str(parent_source), child_peak_rss_bytes=1, files=rows))
        if merlin:
            delivery = intake.read_json(source / "delivery.json")
            delivery.update(job_id_namespace="merlin_run_trial", execution_context=source_execution,
                            node=source_execution["hostname"], circuit=self.context.record(circuit, "recurrent")["circuit"],
                            parent_job_id=str(parent["source_job_id"]), parent_job_id_namespace="slurm",
                            production_episodes=0, source_simulations=0, recorded_utc="fixture-time")
            self.write_json(source / "delivery.json", delivery)
            self.write_json(self.context.circuit(circuit) / "recurrent_acceptance.json", parent)
            return source, parent
        self.write_json(self.context.record_path(circuit, "inputs"), dict(status="COMPLETE_SOURCE_CONSTRUCTION",
            exit_code=0, circuit=dict(config_sha256=config_sha), array_task_id=circuit, array_job_id="624319",
            job_id=str(source_job), parent_job_id=str(parent["source_job_id"]), archive=str(source),
            delivery_sha256=intake.sha256_file(source / "delivery.json")))
        return source, parent

    def merlin_args(self, source, circuit=98):
        delivery = intake.read_json(source / "delivery.json")
        manifest = dict(schema="v1_merlin_input_deliveries_v1", constructions=[dict(
            circuit_index=circuit, archive=str(source), delivery_sha256=intake.sha256_file(source / "delivery.json"),
            execution_context=delivery["execution_context"])])
        path = self.root / "merlin_sources.json"
        self.write_json(path, manifest)
        args = self.args()
        args.construction_manifest = path
        args.construction_manifest_sha256 = intake.sha256_file(path)
        return args

    def test_merlin_input_caller_preserves_namespace_and_writes_only_candidates(self):
        source, _ = self.inputs(98, merlin=True)
        args = self.merlin_args(source)
        with mock.patch.object(intake, "cpu_execution_provenance", return_value=self.execution), \
             mock.patch.object(intake, "memory_evidence", return_value={"fixture": True}), \
             mock.patch.object(intake.ProductionCohort, "load", return_value=self.cohort), \
             mock.patch.object(intake, "accounting_row", side_effect=AssertionError("No Slurm accounting for Merlin")):
            report = intake.run_merlin_intake(args)
        self.assertEqual(report["exit_code"], 0, report)
        acceptance = intake.read_json(args.output / "circuit_098/input_acceptance.json")
        record = intake.read_json(args.output / "circuit_098/inputs/report.json")
        self.assertEqual(acceptance["source_job_id"], "merlin:source-run:source-trial")
        self.assertEqual(acceptance["source_job_id_namespace"], "merlin_run_trial")
        self.assertEqual(acceptance["circuit_index"], 98)
        self.assertEqual(record["execution_context"], acceptance["source_execution_context"])
        self.assertEqual(record["circuit_index"], 98)
        self.assertNotIn("array_job_id", record)
        self.assertNotIn("array_task_id", record)
        self.assertFalse((self.context.circuit(98) / "input_acceptance.json").exists())
        self.assertFalse(self.context.record_path(98, "inputs").exists())

    def test_merlin_manifest_run_and_trial_are_bound_to_delivery(self):
        source, _ = self.inputs(98, merlin=True)
        args = self.merlin_args(source)
        original = intake.read_json(args.construction_manifest)
        for field in ("run_id", "trial_id"):
            with self.subTest(field=field):
                manifest = deepcopy(original)
                manifest["constructions"][0]["execution_context"][field] = "different"
                self.write_json(args.construction_manifest, manifest)
                args.construction_manifest_sha256 = intake.sha256_file(args.construction_manifest)
                context = intake.load_merlin_bound_inputs(args)
                with self.assertRaisesRegex(ValueError, "MERLIN_SOURCE_MANIFEST_DIFFERS"):
                    intake.merlin_input_record(context, manifest["constructions"][0])

    def test_merlin_manifest_cannot_relabel_c98_as_c99(self):
        source, _ = self.inputs(98, merlin=True)
        self.recurrent(99)
        args = self.merlin_args(source, circuit=99)
        context = intake.load_merlin_bound_inputs(args)
        with self.assertRaisesRegex(ValueError, "MERLIN_CIRCUIT_BINDING_DIFFERS"):
            intake.merlin_input_record(context, context.accounting["constructions"][0])

    def test_source_identity_rejects_gpu_merlin_without_legacy_fallback(self):
        record = dict(job_id="merlin:source-run:source-trial", job_id_namespace="merlin_run_trial",
                      execution_context=dict(self.execution, run_id="source-run", trial_id="source-trial", allocated_gpus=1))
        with self.assertRaisesRegex(ValueError, "SOURCE_MERLIN_CPU_IDENTITY_REQUIRED"):
            intake.source_identity(record)
        self.assertEqual(intake.source_identity(dict(job_id="626825")), {"source_job_id": 626825})
        self.assertIs(type(intake.source_identity(dict(job_id="626825"))["source_job_id"]), int)

    def recurrent_acceptance(self, circuit=95, cohort=None):
        return intake.validate_recurrent(self.context, circuit, cohort or self.cohort,
                                         self.execution, self.root / "consumer_scratch")

    def test_recurrent_authenticates_nodes_without_reading_missing_edges(self):
        source, _ = self.recurrent()
        acceptance = self.recurrent_acceptance()
        self.assertFalse((source / "primary/biophysical/v1_edges.h5").exists())
        self.assertFalse(acceptance["edge_archives_rehashed_during_this_intake"])
        self.assertEqual(acceptance["source_job_id"], 628376)
        self.assertEqual(acceptance["acceptance_job"], "merlin:fixture-run")
        self.assertEqual(acceptance["acceptance_step_id"], "merlin:fixture-trial")
        self.assertEqual(acceptance["acceptance_host"], "fixture-merlin-host")
        self.assertGreater(self.cohort.freshness_checks, 0)

    def test_recurrent_changed_node_is_rejected(self):
        source, _ = self.recurrent()
        (source / "primary" / intake.REFERENCE_FILES[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "REFERENCE_DELIVERY_CHANGED"):
            self.recurrent_acceptance()

    def test_recurrent_disconnected_graph_is_rejected(self):
        self.recurrent(connected=False)
        with self.assertRaisesRegex(ValueError, "FULL_COUPLED_RECURRENT_GRAPH_REQUIRED"):
            self.recurrent_acceptance()

    def test_training_seeds_in_dataset_production_are_rejected(self):
        self.recurrent(training_seeds=[41])
        with self.assertRaisesRegex(ValueError, "RECURRENT_SOURCE_QUALIFICATION_FAILED"):
            self.recurrent_acceptance()

    def test_c99_requires_both_raw_and_array_member_identities(self):
        accounting = deepcopy(self.accounting)
        accounting["stdout"] += "\n624318|624318_95|COMPLETED|0:0|10"
        self.assertEqual(intake.accounting_row(accounting, "recurrent", 99, 624318)["array_member"], "624318_99")
        accounting["stdout"] = accounting["stdout"].replace("624318|624318_99|COMPLETED", "624318|624318_99|FAILED")
        with self.assertRaisesRegex(ValueError, "SOURCE_ACCOUNTING_NOT_COMPLETE"):
            intake.accounting_row(accounting, "recurrent", 99, 624318)

    def test_pending_input_c98_is_outside_authorized_intake(self):
        with self.assertRaisesRegex(ValueError, "TASK_OUTSIDE_NINE_CONSTRUCTIONS"):
            intake.accounting_row(self.accounting, "inputs", 98, 626828)

    def test_input_acceptance_hashes_all_174_payloads(self):
        source, parent = self.inputs()
        acceptance = intake.validate_inputs(self.context, 95, self.cohort, self.execution, parent)
        self.assertEqual(acceptance["verified_files"], 174)
        self.assertEqual(acceptance["files_compared"], 20)
        self.assertEqual(acceptance["verified_bytes"], sum(row["bytes"] for row in intake.read_json(source / "delivery.json")["files"]))
        self.assertNotIn("acceptance_host", acceptance)
        self.assertGreater(self.cohort.freshness_checks, 0)

    def test_input_nonnative_manifest_payload_corruption_is_rejected(self):
        source, parent = self.inputs()
        (source / "extra/payload_000.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "INPUT_DELIVERY_PAYLOAD_CHANGED:extra/"):
            intake.validate_inputs(self.context, 95, self.cohort, self.execution, parent)

    def test_input_twenty_file_comparison_cannot_duplicate_a_path(self):
        _, parent = self.inputs(comparison_complete=False)
        with self.assertRaisesRegex(ValueError, "COMPLETE_TWENTY_FILE_INPUT_REPLAY_REQUIRED"):
            intake.validate_inputs(self.context, 95, self.cohort, self.execution, parent)

    def test_input_requires_matching_accepted_parent(self):
        _, parent = self.inputs()
        parent["consumer_binding"]["delivery_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ACCEPTED_RECURRENT_PARENT_REQUIRED"):
            intake.validate_inputs(self.context, 95, self.cohort, self.execution, parent)

    def test_input_cohort_freshness_is_required_before_acceptance(self):
        _, parent = self.inputs()
        with self.assertRaisesRegex(ValueError, "FIXTURE_COHORT_CHANGED"):
            intake.validate_inputs(self.context, 95, CohortFixture(stale=True), self.execution, parent)

    def test_existing_acceptance_bytes_are_preserved(self):
        source, parent = self.inputs()
        acceptance = intake.validate_inputs(self.context, 95, self.cohort, self.execution, parent)
        canonical = self.context.circuit(95) / "input_acceptance.json"
        acceptance["acceptance_job"] = "original-slurm-fixture"
        original = json.dumps(acceptance, indent=3) + "\n"
        canonical.write_text(original)
        generated = dict(acceptance, acceptance_job="merlin:different-run")
        destination = self.root / "isolated/input_acceptance.json"
        saved, reused = intake.save_acceptance(canonical, destination, generated, "inputs")
        self.assertTrue(reused)
        self.assertEqual(saved["acceptance_job"], "original-slurm-fixture")
        self.assertEqual(canonical.read_text(), original)
        self.assertEqual(destination.read_text(), original)

    def test_staging_excludes_recurrent_edges_and_future_reference_inputs(self):
        for circuit in range(94, 100):
            self.recurrent(circuit)
        for circuit in range(94, 98):
            self.inputs(circuit)
        for name in ("frozen_campaign.yaml", "qualification.json", "identities.csv"):
            (self.q / name).write_text("fixture metadata")
        files = intake.staging_files(self.context)
        self.assertFalse(any("/recurrent_" in path and "v1_edges.h5" in path for path in files))
        self.assertFalse(any("reference_execution" in path or "reference_preparation" in path for path in files))
        self.assertIn(str(self.q / "identities.json"), files)
        self.assertIn(str(self.sources / "inputs_95/extra/payload_000.bin"), files)

    def test_run_rejects_slurm_execution_without_creating_output(self):
        with mock.patch.object(intake, "cpu_execution_provenance", return_value=dict(backend="slurm")):
            with self.assertRaisesRegex(ValueError, "ACTUAL_MERLIN_EXECUTION_REQUIRED"):
                intake.run_intake(self.args())
        self.assertFalse((self.root / "result").exists())

    def test_run_rejects_output_under_canonical_root(self):
        with mock.patch.object(intake, "cpu_execution_provenance", return_value=self.execution), \
                mock.patch.object(intake, "memory_evidence", return_value={"fixture": True}):
            with self.assertRaisesRegex(ValueError, "ISOLATED_OUTPUT_REQUIRED"):
                intake.run_intake(self.args(self.q / "must_not_exist"))
        self.assertFalse((self.q / "must_not_exist").exists())

    def test_run_preserves_order_and_never_prepares_reference_jobs(self):
        self.recurrent(94)
        calls = []

        def recurrent(context, circuit, cohort, execution, scratch):
            calls.append(("recurrent", circuit))
            return dict(status="PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY", source_job_id=circuit)

        def inputs(context, circuit, cohort, execution, parent):
            calls.append(("inputs", circuit))
            return dict(status="PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY", source_job_id=circuit)

        with mock.patch.object(intake, "cpu_execution_provenance", return_value=self.execution), \
                mock.patch.object(intake, "memory_evidence", return_value={"fixture": True}), \
                mock.patch.object(intake.ProductionCohort, "load", return_value=self.cohort), \
                mock.patch.object(intake, "validate_recurrent", side_effect=recurrent), \
                mock.patch.object(intake, "validate_inputs", side_effect=inputs):
            report = intake.run_intake(self.args())
        self.assertEqual(calls, [(stage, circuit) for stage, circuit, job in intake.TASKS])
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(len(report["acceptances"]), 9)
        self.assertEqual(report["canonical_acceptances_written"], 0)
        self.assertEqual(report["future_reference_preparations"], 0)
        self.assertFalse(any(path.name in {"reference_execution", "reference_preparation"}
                             for path in (self.root / "result").rglob("*")))
        self.assertFalse((self.q / "circuit_095/recurrent_acceptance.json").exists())


if __name__ == "__main__":
    unittest.main()

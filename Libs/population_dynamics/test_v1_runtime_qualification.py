"""Synthetic admission fixtures only; execute on the authorized Merlin test job."""
from copy import deepcopy
import csv
from dataclasses import FrozenInstanceError, replace
import io
import json
import tarfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from . import benchmark_production as production
from . import circuit_data
from . import v1_runtime_qualification as runtime
from .benchmark_dataset import read_episode
from .circuit_data import CircuitArchiveWriter, CircuitDataset
from .provenance import sha256_file
from .test_circuit_data import binding as tiny_binding, episode as tiny_episode, units as tiny_units
from .test_benchmark_production import fixture_configuration


def document(value):
    payload = json.dumps(value, sort_keys=True)
    return {"payload": payload, "sha256": runtime.digest_bytes(payload.encode())}


def replace_document(artifact, role, update):
    value = runtime.decode_document(artifact["documents"][role])
    update(value)
    artifact["documents"][role] = document(value)
    seal(artifact)


def seal(artifact):
    manifest = {}
    for role, name in artifact["evidence_paths"].items():
        doc = artifact["documents"][role]
        manifest[name] = {"bytes": len(doc["payload"].encode()), "sha256": doc["sha256"]}
    for label in ("primary", "independent_replay"):
        for index in range(10):
            for filename in ("cortical_state.npz", "cortical_spikes.npz"):
                manifest[f"results/run/{label}/condition_{index:02d}/{filename}"] = {
                    "bytes": 1, "sha256": "e" * 64}
    artifact["documents"]["return_manifest"] = document(manifest)
    artifact["verified_return"].update(manifest_sha256=artifact["documents"]["return_manifest"]["sha256"],
                                       verified_files=len(manifest))


@pytest.fixture
def proof(monkeypatch):
    # Only this disposable fixture's case-file digest differs from the production constant.
    backend = {"library_sha256": runtime.ORIGINAL_LIBRARY_SHA256,
               "runtime_code": {name: "a" * 64 for name in runtime.RUNTIME_NAMES},
               "component_rtol": 1e-12, "component_atol": 1e-12}
    original = document({"source_recipes": {"glif_cuda": backend}})
    cases = {"schema": "v1_native_block_comparison_v1", "scenario_id": "v1_visual",
             "circuit_index": 0, "block_index": 2, "cases": []}
    for i in range(10):
        identity = dict(runtime.STREAMS, scenario_id="v1_visual", circuit_index=0,
                        block_index=2, condition_index=i, split="train",
                        episode_id=f"v1_visual_circuit_000_block_002_condition_{i:02d}",
                        sample_unit="complete_mechanistic_episode")
        cases["cases"].append({"identity": identity, "native_job_id": 100 + i,
                               "native_archive": f"/native/{i}", "native_report_sha256": f"{i:064x}"})
    case_doc = document(cases)
    monkeypatch.setattr(runtime, "CASES_SHA256", case_doc["sha256"])
    runtime._validated_payload.cache_clear()
    docs = {"original_qualification": original, "cases": case_doc}
    docs["compiler"] = document({
        "publication_status": "PASS", "exit_code": 0, "target_arch": "sm_80",
        "library_sha256": runtime.CANDIDATE_LIBRARY_SHA256,
        "original_library_sha256": runtime.ORIGINAL_LIBRARY_SHA256,
        "source_sha256": runtime.CUDA_SOURCE_SHA256, "nvcc_version": "V12.8.61", "run_sid": "compiler"})
    docs["snapshot"] = document({
        "status": "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT", "native_report_sha256": f"{0:064x}",
        "native_reference": "/native/0", "expected_source_batch_sha256": runtime.SOURCE_BATCH_SHA256,
        "snapshot": {"source_batch_sha256": runtime.SOURCE_BATCH_SHA256, "rows": 82246780},
        "units": 230924, "native_simulation_calls": 0})
    executions = []
    for number, label in enumerate(("primary", "independent_replay")):
        executions.append({"label": label, "process_id": number + 1, "exit_code": 0, "passed": True,
                           "legacy_compatibility_id": label})
        rows = []
        for i, case in enumerate(cases["cases"]):
            values = {"all_spike_pairs_exact": True, "all_unit_counts_exact": True,
                      "terminal_voltage_within_component_tolerance": True}
            rows.append(dict(values, condition_index=i, identity=case["identity"],
                             native_job_id=case["native_job_id"], status=runtime.NATIVE_PASS))
            docs[f"{label}_condition_{i:02d}"] = document(dict(
                values, job_id=label, status=runtime.NATIVE_PASS, units=230924, duration_ms=3000.,
                runtime_code=backend["runtime_code"], component_rtol=1e-12, component_atol=1e-12,
                cuda_execution={"library_sha256": runtime.CANDIDATE_LIBRARY_SHA256},
                source_bindings={"native_job_id": case["native_job_id"],
                                 "files": {"primary/report.json": {"sha256": case["native_report_sha256"]}}}))
        docs[label] = document({"job_id": label, "status": runtime.BLOCK_PASS, "planned_conditions": 10,
                                "passed_conditions": 10, "failed_conditions": 0, "cases": rows,
                                "runtime_code": backend["runtime_code"], "cases_sha256": case_doc["sha256"],
                                "coupled_units": 230924, "source_connection_rows": 82246780,
                                "gpu_device_preflight": {"library_sha256": runtime.CANDIDATE_LIBRARY_SHA256}})
    replay = {"planned_conditions": 10, "passed_conditions": 10, "failed_conditions": 0,
              "conditions": [{"condition_index": i, "passed": True,
                              "checks": {key: True for key in runtime.REPLAY_CHECKS},
                              "metrics": {"terminal_values_exact": False,
                                          "terminal_maximum_absolute_difference_mV": 2.13e-14}}
                             for i in range(10)]}
    family = {"status": runtime.FAMILY_PASS, "exit_code": 0, "run_sid": "run", "trial_sid": "trial",
              "output": "/results/run", "compute_capability": [8, 0],
              "library_sha256": runtime.CANDIDATE_LIBRARY_SHA256,
              "source_sha256": runtime.CUDA_SOURCE_SHA256, "compiler_run_sid": "compiler",
              "units": 230924, "duration_ms": 3000., "passed_native_conditions": 20,
              "passed_executions": 2, "component_rtol": 1e-12, "component_atol": 1e-12,
              "production_qualified": False, "scientific_qualification": False,
              "new_benchmark_episodes": 0, "executions": executions, "independent_replay": replay}
    docs.update(family_report=document(family), replay_comparison=document(replay),
                receipt=document(dict(family, persisted=True, result_sha256="b" * 64, result_bytes=123)))
    artifact = {"schema": runtime.SCHEMA, "status": "PASS",
                "original_qualification_sha256": original["sha256"], "original_backend": backend,
                "backend": dict(backend, library_sha256=runtime.CANDIDATE_LIBRARY_SHA256,
                                compute_capability=[8, 0]),
                "documents": docs,
                "evidence_paths": {role: "results/run/" + role + ".json" for role in docs
                                   if role not in {"original_qualification", "receipt", "return_manifest"}},
                "verified_return": {"archive_sha256": "b" * 64, "archive_bytes": 123}}
    seal(artifact)
    yield artifact, original["sha256"], backend
    runtime._validated_payload.cache_clear()


def test_complete_proof_accepts_non_bitwise_within_budget(proof):
    artifact, sha, backend = proof
    assert runtime.validate_artifact(artifact, sha, backend)["library_sha256"] == runtime.CANDIDATE_LIBRARY_SHA256


def test_default_sm89_recipe_requires_no_runtime_proof(proof):
    _, _, backend = proof
    cohort = SimpleNamespace(qualification={"source_recipes": {"glif_cuda": backend}})
    assert production.glif_backend_qualification(cohort) == backend


@pytest.mark.parametrize("role,change", [
    ("primary", lambda value: value.update(passed_conditions=9)),
    ("primary_condition_04", lambda value: value.update(units=230923)),
    ("independent_replay_condition_09", lambda value: value.update(component_atol=1e-11)),
    ("compiler", lambda value: value.update(source_sha256="f" * 64)),
    ("snapshot", lambda value: value.update(expected_source_batch_sha256="f" * 64)),
    ("replay_comparison", lambda value: value["conditions"][2]["checks"].pop("same_reference_binding")),
])
def test_rejects_rebound_partial_or_mismatched_evidence(proof, role, change):
    artifact, sha, backend = proof
    replace_document(artifact, role, change)
    with pytest.raises(ValueError):
        runtime.validate_artifact(artifact, sha, backend)


def test_rejects_document_tamper_and_wrong_original(proof):
    artifact, sha, backend = proof
    bad = deepcopy(artifact)
    bad["documents"]["compiler"]["payload"] += " "
    with pytest.raises(ValueError):
        runtime.validate_artifact(bad, sha, backend)
    with pytest.raises(ValueError):
        runtime.validate_artifact(artifact, "f" * 64, backend)


def episode_fixture(proof):
    artifact, sha, _ = proof
    proof_doc = document(artifact)
    binding = {"scope": "production", "scenario_id": "v1_visual",
               "source_qualification_sha256": sha, "identity_table_sha256": "c" * 64}
    identity = {"episode_id": "production_episode", "circuit_id": "production_circuit",
                "block_id": "production_block", "split": "train", **runtime.STREAMS}
    report = {"runtime_code": artifact["backend"]["runtime_code"],
              "cuda_execution": {"library_sha256": runtime.CANDIDATE_LIBRARY_SHA256},
              "circuit_bindings": {"files": {"primary/report.json": {"sha256": "d" * 64}}},
              "source_bindings": {"files": {"primary/report.json": {"sha256": "d" * 64}}}}
    report_doc = document(report)
    admission = {"runtime_qualification_sha256": proof_doc["sha256"], "execution_backend": "glif_cuda",
                 "source_qualification_sha256": sha, "identity_table_sha256": "c" * 64,
                 "cohort_identity": identity, "circuit_reference_report_sha256": "d" * 64,
                 "source_documents": {role: report_doc for role in ("execution", "replay", "circuit_validation")},
                 **{key: report_doc["sha256"] for key in
                    ("execution_report_sha256", "replay_report_sha256", "circuit_validation_sha256")}}
    metadata = {key: identity[key] for key in ("episode_id", "circuit_id", "block_id", "split")}
    metadata.update(seeds=dict(runtime.STREAMS), source_record={"production_source_admission": admission})
    return binding, {proof_doc["sha256"]: proof_doc}, metadata, {
        "runtime_qualification_sha256": proof_doc["sha256"]}


def test_registered_episode_requires_matching_identity_and_catalog_proof(proof):
    binding, registry, metadata, row = episode_fixture(proof)
    runtime.validate_episode(binding, registry, metadata, row)
    with pytest.raises(ValueError):
        runtime.validate_episode(binding, {}, metadata, row)
    with pytest.raises(ValueError):
        runtime.validate_episode(binding, registry, dict(metadata, circuit_id="other"), row)
    with pytest.raises(ValueError):
        runtime.validate_episode(binding, registry, metadata, {})


def test_old_episode_and_empty_registry_remain_compatible():
    runtime.validate_registry({"scope": "production"}, {})
    runtime.validate_episode({"scope": "production"}, {}, {"source_record": {}}, {})



def _known_candidate_admission(library=runtime.CANDIDATE_LIBRARY_SHA256):
    return {"execution_backend": "glif_cuda", "source_documents": {
        "execution": document({"cuda_execution": {"library_sha256": library}})}}


def _tiny_archive(root):
    archive_binding = dict(tiny_binding(), scenario_id="v1_visual")
    with CircuitArchiveWriter(root, archive_binding) as writer:
        writer.register_circuit(tiny_units())
        writer.add(tiny_episode())
    return archive_binding


def _set_stored_admission(root, admission):
    path = root / "episodes_000000.h5"
    with h5py.File(path, "r+") as source:
        group = source["episode_a"]
        metadata = json.loads(group.attrs["metadata_json"])
        metadata["source_record"]["production_source_admission"] = admission
        group.attrs["metadata_json"] = json.dumps(metadata)
    catalog_path = root / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["shards"][0]["sha256"] = sha256_file(path)
    catalog_path.write_text(json.dumps(catalog))


def test_writer_rejects_known_candidate_with_both_runtime_references_missing(tmp_path):
    binding = dict(tiny_binding(), scenario_id="v1_visual")
    with CircuitArchiveWriter(tmp_path, binding) as writer:
        writer.register_circuit(tiny_units())
        episode = tiny_episode()
        invalid = replace(episode, source_record=dict(
            episode.source_record, production_source_admission=_known_candidate_admission()))
        with pytest.raises(ValueError, match="KNOWN_CANDIDATE_REQUIRES_RUNTIME_IDENTITY"):
            writer.add(invalid)
        assert writer.ids == set()
        assert writer.catalog["complete_episodes"] == 0
        assert writer.handle is None


@pytest.mark.parametrize("reader", ["raw_episode", "public_complete_episode"])
def test_actual_readers_reject_stripped_runtime_references(tmp_path, reader):
    _tiny_archive(tmp_path)
    _set_stored_admission(tmp_path, _known_candidate_admission())
    dataset = CircuitDataset(tmp_path, splits=("train",))
    with pytest.raises(ValueError, match="KNOWN_CANDIDATE_REQUIRES_RUNTIME_IDENTITY"):
        if reader == "raw_episode":
            read_episode(dataset, dataset.episodes["episode_a"])
        else:
            dataset.complete_episode("episode_a", bin_ticks=2)


@pytest.mark.parametrize("library", [runtime.ORIGINAL_LIBRARY_SHA256, "f" * 64])
def test_actual_default_writer_and_readers_keep_prior_backend_support(tmp_path, library):
    binding = dict(tiny_binding(), scenario_id="v1_visual")
    episode = tiny_episode()
    valid = replace(episode, source_record=dict(
        episode.source_record, production_source_admission=_known_candidate_admission(library)))
    with CircuitArchiveWriter(tmp_path, binding) as writer:
        writer.register_circuit(tiny_units())
        writer.add(valid)
    dataset = CircuitDataset(tmp_path, splits=("train",))
    assert read_episode(dataset, dataset.episodes["episode_a"]).episode_id == "episode_a"
    assert dataset.complete_episode("episode_a", bin_ticks=2)["episode_id"] == "episode_a"


def _unpublish_closed_shard(root):
    catalog_path = root / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    catalog.update(shards=[], complete_episodes=0)
    catalog_path.write_text(json.dumps(catalog))
    return catalog_path.read_bytes()


def test_closed_shard_recovery_rejects_runtime_omission_before_mutation(tmp_path):
    binding = _tiny_archive(tmp_path)
    _set_stored_admission(tmp_path, _known_candidate_admission())
    original_catalog = _unpublish_closed_shard(tmp_path)
    writer = CircuitArchiveWriter.__new__(CircuitArchiveWriter)
    with pytest.raises(ValueError, match="KNOWN_CANDIDATE_REQUIRES_RUNTIME_IDENTITY"):
        writer.__init__(tmp_path, binding, resume=True)
    assert writer.ids == set()
    assert writer.circuit_splits == {}
    assert writer.catalog["complete_episodes"] == 0
    assert writer.catalog["shards"] == []
    assert (tmp_path / "dataset.json").read_bytes() == original_catalog
    assert (tmp_path / "episodes_000000.h5").is_file()


def test_closed_shard_recovery_keeps_genuine_legacy_archive(tmp_path):
    binding = _tiny_archive(tmp_path)
    _unpublish_closed_shard(tmp_path)
    with CircuitArchiveWriter(tmp_path, binding, resume=True) as writer:
        assert writer.ids == {"episode_a"}
        assert writer.catalog["complete_episodes"] == 1
    dataset = CircuitDataset(tmp_path, splits=("train",))
    assert read_episode(dataset, dataset.episodes["episode_a"]).episode_id == "episode_a"



def test_completed_episode_recovers_sm80_closed_shard_with_selected_proof(
        tmp_path, proof, monkeypatch):
    # Substitute only the full campaign admission boundary. Runtime proof,
    # writer, closed-shard recovery and public readback remain real.
    artifact, _, backend = proof
    original = runtime.decode_document(artifact["documents"]["original_qualification"])
    original["source_revision"] = "test-source"
    original_doc = document(original)
    artifact["documents"]["original_qualification"] = original_doc
    artifact["original_qualification_sha256"] = original_doc["sha256"]
    binding, registry, metadata, _ = episode_fixture(
        (artifact, original_doc["sha256"], backend))
    runtime_doc = next(iter(registry.values()))
    identity = metadata["source_record"]["production_source_admission"]["cohort_identity"]
    identity.update(scenario_id="v1_visual", circuit_index=0, condition_index=0)
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text(original_doc["payload"])
    identity_table = tmp_path / "identities.csv"
    with identity_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(identity))
        writer.writeheader()
        writer.writerow(identity)
    table_sha = sha256_file(identity_table)
    campaign = {"scenarios": {"v1_visual": {"task_protocol": {
        "condition_duration_ticks": [10], "dt_ms": .1}}}}
    identity_table.with_suffix(".json").write_text(json.dumps({
        "resolved_campaign": campaign,
        "identity_table_sha256": table_sha,
        "source_qualification_sha256": original_doc["sha256"]}))
    monkeypatch.setattr(circuit_data, "production_prerequisites", lambda *args: [])
    monkeypatch.setattr(circuit_data, "episode_identities",
                        lambda *args, **kwargs: iter([identity]))
    monkeypatch.setattr(production.ProductionCohort, "require_current", lambda self: None)
    cohort = production.ProductionCohort(
        campaign, "v1_visual", qualification_path, identity_table, original,
        original_doc["sha256"], table_sha, {identity["episode_id"]: identity},
        runtime_document=runtime_doc)
    admission = metadata["source_record"]["production_source_admission"]
    admission["identity_table_sha256"] = table_sha
    source = tiny_episode()
    unit_table = replace(tiny_units(), circuit_id=identity["circuit_id"],
                         unit_ids=np.arange(100000), populations=np.full(100000, "E"),
                         field_index=np.zeros(100000, dtype=np.int64), field_names=("E",),
                         coordinates=None, coordinate_units=None)
    record = dict(source.source_record, core_unit_count=100000,
                  production_source_admission=admission,
                  cohort_identity=identity, production_declaration=cohort.declaration(identity),
                  native_seed_streams=dict(runtime.STREAMS))
    episode = replace(source, episode_id=identity["episode_id"],
                      circuit_id=identity["circuit_id"], block_id=identity["block_id"],
                      seeds=dict(runtime.STREAMS), source_record=record)
    output = tmp_path / "archive"
    with cohort.writer(output, "test-source") as writer:
        writer.register_circuit(unit_table)
        writer.add(episode)
    shard_sha = sha256_file(output / "episodes_000000.h5")
    _unpublish_closed_shard(output)
    catalog_path = output / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    catalog.pop("runtime_qualifications")
    catalog_path.write_text(json.dumps(catalog))
    verified = []

    def verify_recorded_runtime(selected_cohort, actual_identity, actual_admission):
        assert selected_cohort.runtime_document == runtime_doc
        assert actual_identity == identity
        assert actual_admission == admission
        verified.append(selected_cohort.runtime_document["sha256"])

    monkeypatch.setattr(production, "verify_source_admission", verify_recorded_runtime)
    result = production.completed_episode(cohort, identity, output, resume=True)
    assert result["status"] == "REUSED_COMMITTED_PRODUCTION_EPISODE"
    assert result["complete_episodes"] == 1
    assert result["new_source_simulations"] == 0
    assert verified == [runtime_doc["sha256"]]
    recovered = json.loads(catalog_path.read_text())
    assert recovered["complete_episodes"] == 1
    assert recovered["runtime_qualifications"] == registry
    assert recovered["shards"][0]["episodes"][0]["runtime_qualification_sha256"] == runtime_doc["sha256"]
    assert sha256_file(output / "episodes_000000.h5") == shard_sha
    dataset = CircuitDataset(output, splits=("train",))
    assert read_episode(dataset, dataset.episodes[identity["episode_id"]]).seeds == dict(runtime.STREAMS)


def test_builder_rejects_duplicate_members_even_with_matching_archive_hash(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    archive_path = tmp_path / "returned.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for _ in range(2):
            info = tarfile.TarInfo("merlin_return_manifest.json")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"{}"))
    identity = runtime._file_identity(archive_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"result_sha256": identity["sha256"],
                                       "result_bytes": identity["bytes"]}))
    with pytest.raises(ValueError, match="RETURN_ARCHIVE_DUPLICATE_MEMBER"):
        runtime.prepare(tmp_path / "unused_qualification.json", tmp_path / "unused_result",
                        receipt_path, archive_path, manifest_path, tmp_path / "must_not_exist.json")
    assert not (tmp_path / "must_not_exist.json").exists()



def test_real_cohort_load_binds_runtime_without_mutating_frozen_cohort(proof, tmp_path):
    artifact, _, backend = proof
    config, qualification, table = fixture_configuration(tmp_path / "cohort", "v1_visual")
    qualified = json.loads(qualification.read_text())
    qualified["source_recipes"]["glif_cuda"] = backend
    original = document(qualified)
    qualification.write_text(original["payload"])
    frozen_path = table.with_suffix(".json")
    frozen = json.loads(frozen_path.read_text())
    frozen["source_qualification_sha256"] = original["sha256"]
    frozen_path.write_text(json.dumps(frozen))
    artifact = deepcopy(artifact)
    artifact["original_qualification_sha256"] = original["sha256"]
    artifact["documents"]["original_qualification"] = original
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(artifact))
    before = (qualification.read_bytes(), table.read_bytes(), frozen_path.read_bytes())

    cohort = production.ProductionCohort.load(
        config, "v1_visual", qualification, table, runtime_qualification=runtime_path)

    cohort.require_current()
    assert cohort.runtime_path == runtime_path
    assert cohort.runtime_document == runtime.source_document(runtime_path)
    assert production.glif_backend_qualification(cohort)["library_sha256"] == runtime.CANDIDATE_LIBRARY_SHA256
    assert before == (qualification.read_bytes(), table.read_bytes(), frozen_path.read_bytes())
    with pytest.raises(FrozenInstanceError):
        cohort.runtime_path = None

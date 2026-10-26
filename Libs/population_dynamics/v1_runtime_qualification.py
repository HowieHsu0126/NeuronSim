"""Explicit sm80 portability evidence; original source/cohort identities stay fixed."""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tarfile

SCHEMA = "v1_glif_runtime_qualification_v1"
CASES_SHA256 = "a6fad79c884c66fe3e04d9da5a5ffb383c5419cab04ab6b846a89f41d7a5959f"
SOURCE_BATCH_SHA256 = "09b8dd29bd6c30117c2ed22fdfd0b56d0b45444e405334bf4ca178911f25db17"
CUDA_SOURCE_SHA256 = "18014a01f197dafa8bc4033cbad81e05283810e1109a5aaf530b80d5837ce8c7"
CANDIDATE_LIBRARY_SHA256 = "05b5b20444446c137ba6b9a482bb811fabed47acb5d80427661c540b88967831"
ORIGINAL_LIBRARY_SHA256 = "76a11a0a3e6de5f4d3a7e16cdb2ddd37276e745bdd2c6959392e75414a3ac2ae"
RUNTIME_NAMES = {"v1_glif_network.py", "v1_glif_component_emulator.py",
                 "v1_glif_cuda.py", "v1_nest_time_semantics.py"}
NATIVE_PASS = "PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC"
BLOCK_PASS = "PASS_COMPLETE_TEN_CONDITION_NATIVE_GLIF_CUDA_BLOCK"
FAMILY_PASS = "PASS_FROZEN_C0_B2_NATIVE_BATCH_AND_INDEPENDENT_REPLAY"
REPLAY_CHECKS = {"fresh_output_directories", "native_comparisons_passed",
                 "frozen_tolerances_match", "same_reference_binding",
                 "cortical_state.npz_keys", "cortical_state.npz_dtype_shape",
                 "cortical_spikes.npz_keys", "cortical_spikes.npz_dtype_shape",
                 "state_identity_initial_counts_exact", "finite_terminal_values",
                 "terminal_within_frozen_budget", "spike_pairs_exact_after_sort"}
STREAMS = {"connectivity_seed": 100000000, "initial_state_seed": 110000002,
           "background_seed": 120000002, "stimulus_seed": 130000002}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("GLIF_RUNTIME_" + message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_document(path: Path) -> dict:
    payload = path.read_bytes()
    return {"payload": payload.decode("utf-8"), "sha256": digest_bytes(payload)}


def decode_document(document: dict) -> dict:
    payload = document.get("payload")
    require(isinstance(payload, str) and
            digest_bytes(payload.encode()) == document.get("sha256"), "DOCUMENT_HASH_DIFFERS")
    value = json.loads(payload)
    require(isinstance(value, dict), "DOCUMENT_OBJECT_REQUIRED")
    return value


def _all_true(checks: dict) -> bool:
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def validate_artifact(artifact: dict, original_sha256: str, original_backend: dict) -> dict:
    require(artifact.get("schema") == SCHEMA and artifact.get("status") == "PASS"
            and artifact.get("original_qualification_sha256") == original_sha256,
            "ORIGINAL_QUALIFICATION_BINDING_DIFFERS")
    require(original_backend.get("library_sha256") == ORIGINAL_LIBRARY_SHA256
            and set(original_backend.get("runtime_code", {})) == RUNTIME_NAMES
            and original_backend.get("component_rtol") == original_backend.get("component_atol") == 1e-12,
            "FROZEN_ORIGINAL_BACKEND_DIFFERS")
    expected = {key: original_backend[key] for key in
                ("runtime_code", "library_sha256", "component_rtol", "component_atol")}
    require(artifact.get("original_backend") == expected, "FROZEN_BACKEND_BINDING_DIFFERS")
    backend = artifact.get("backend", {})
    require(backend.get("runtime_code") == expected["runtime_code"]
            and backend.get("library_sha256") == CANDIDATE_LIBRARY_SHA256
            and backend.get("component_rtol") == backend.get("component_atol") == 1e-12
            and backend.get("compute_capability") == [8, 0], "CANDIDATE_BACKEND_DIFFERS")
    documents = artifact.get("documents", {})
    required = {"original_qualification", "compiler", "cases", "snapshot", "family_report", "receipt",
                "return_manifest", "primary", "independent_replay", "replay_comparison"}
    required.update(f"{label}_condition_{i:02d}"
                    for label in ("primary", "independent_replay") for i in range(10))
    require(set(documents) == required, "COMPLETE_EVIDENCE_DOCUMENTS_REQUIRED")
    decoded = {name: decode_document(value) for name, value in documents.items()}
    require(documents["original_qualification"]["sha256"] == original_sha256
            and decoded["original_qualification"]["source_recipes"]["glif_cuda"] == original_backend,
            "ORIGINAL_SOURCE_DOCUMENT_DIFFERS")
    returned = decoded["return_manifest"]
    for role in required - {"original_qualification", "receipt", "return_manifest"}:
        name = artifact.get("evidence_paths", {}).get(role)
        require(name in returned and returned[name]["sha256"] == documents[role]["sha256"]
                and returned[name]["bytes"] == len(documents[role]["payload"].encode()),
                "RETURNED_DOCUMENT_IDENTITY_DIFFERS")
    compiler = decoded["compiler"]
    require(compiler.get("publication_status") == "PASS" and compiler.get("exit_code") == 0
            and compiler.get("library_sha256") == CANDIDATE_LIBRARY_SHA256
            and compiler.get("original_library_sha256") == ORIGINAL_LIBRARY_SHA256
            and compiler.get("source_sha256") == CUDA_SOURCE_SHA256
            and compiler.get("target_arch") == "sm_80" and "V12.8.61" in compiler.get("nvcc_version", ""),
            "COMPILER_IDENTITY_DIFFERS")
    cases = decoded["cases"]
    require(documents["cases"]["sha256"] == CASES_SHA256
            and cases.get("schema") == "v1_native_block_comparison_v1"
            and cases.get("scenario_id") == "v1_visual"
            and cases.get("circuit_index") == 0 and cases.get("block_index") == 2
            and len(cases.get("cases", [])) == 10, "FROZEN_NATIVE_FAMILY_DIFFERS")
    for index, case in enumerate(cases["cases"]):
        identity = case["identity"]
        require(identity.get("condition_index") == index and identity.get("circuit_index") == 0
                and identity.get("block_index") == 2 and identity.get("split") == "train"
                and identity.get("episode_id") == f"v1_visual_circuit_000_block_002_condition_{index:02d}"
                and identity.get("sample_unit") == "complete_mechanistic_episode"
                and all(identity.get(k) == v for k, v in STREAMS.items()),
                "FROZEN_NATIVE_CONDITION_DIFFERS")
    snapshot = decoded["snapshot"]
    require(snapshot.get("status") == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
            and snapshot.get("native_report_sha256") == cases["cases"][0]["native_report_sha256"]
            and snapshot.get("native_reference") == cases["cases"][0]["native_archive"]
            and snapshot.get("expected_source_batch_sha256") == SOURCE_BATCH_SHA256
            and snapshot.get("snapshot", {}).get("source_batch_sha256") == SOURCE_BATCH_SHA256
            and snapshot.get("snapshot", {}).get("rows") == 82246780
            and snapshot.get("units") == 230924 and snapshot.get("native_simulation_calls") == 0,
            "FROZEN_SNAPSHOT_DIFFERS")
    family, receipt = decoded["family_report"], decoded["receipt"]
    require(family.get("status") == receipt.get("status") == FAMILY_PASS
            and family.get("exit_code") == receipt.get("exit_code") == 0
            and family.get("run_sid") == receipt.get("run_sid")
            and family.get("trial_sid") == receipt.get("trial_sid")
            and family.get("run_sid") and family.get("trial_sid")
            and family.get("compute_capability") == [8, 0]
            and family.get("library_sha256") == CANDIDATE_LIBRARY_SHA256
            and family.get("source_sha256") == CUDA_SOURCE_SHA256
            and family.get("compiler_run_sid") == compiler.get("run_sid")
            and family.get("units") == 230924 and family.get("duration_ms") == 3000.0
            and family.get("passed_native_conditions") == 20
            and family.get("passed_executions") == 2
            and family.get("component_rtol") == family.get("component_atol") == 1e-12
            and family.get("production_qualified") is False
            and family.get("scientific_qualification") is False
            and family.get("new_benchmark_episodes") == 0, "COMPLETE_NATIVE_QUALIFICATION_REQUIRED")
    require(all(receipt.get(key) == value for key, value in family.items()),
            "RECEIPT_RESULT_REPORT_DIFFERS")
    prefix = family["output"].lstrip("/") + "/"
    for label in ("primary", "independent_replay"):
        for index in range(10):
            for filename in ("cortical_state.npz", "cortical_spikes.npz"):
                name = prefix + f"{label}/condition_{index:02d}/" + filename
                require(name in returned and returned[name]["bytes"] > 0
                        and len(returned[name]["sha256"]) == 64,
                        "COMPLETE_RETURNED_ARRAYS_REQUIRED")
    verification = artifact.get("verified_return", {})
    require(receipt.get("persisted") is True and verification.get("archive_sha256") == receipt.get("result_sha256")
            and verification.get("archive_bytes") == receipt.get("result_bytes")
            and verification.get("manifest_sha256") == documents["return_manifest"]["sha256"]
            and verification.get("verified_files") == len(decoded["return_manifest"])
            and verification.get("verified_files", 0) > 0, "VERIFIED_RETURN_REQUIRED")
    executions = family.get("executions", [])
    require([row.get("label") for row in executions] == ["primary", "independent_replay"]
            and len({row.get("process_id") for row in executions}) == 2
            and len({row.get("legacy_compatibility_id") for row in executions}) == 2,
            "INDEPENDENT_EXECUTIONS_REQUIRED")
    for label, execution in zip(("primary", "independent_replay"), executions):
        require(execution.get("exit_code") == 0 and execution.get("passed") is True,
                "FAILED_NATIVE_EXECUTION")
        aggregate = decoded[label]
        require(aggregate.get("status") == BLOCK_PASS
                and str(aggregate.get("job_id")) == str(execution["legacy_compatibility_id"])
                and aggregate.get("planned_conditions") == 10
                and aggregate.get("passed_conditions") == 10 and aggregate.get("failed_conditions") == 0
                and aggregate.get("runtime_code") == expected["runtime_code"]
                and aggregate.get("cases_sha256") == CASES_SHA256
                and aggregate.get("coupled_units") == 230924
                and aggregate.get("source_connection_rows") == 82246780
                and aggregate.get("gpu_device_preflight", {}).get("library_sha256") == CANDIDATE_LIBRARY_SHA256
                and [row.get("condition_index") for row in aggregate.get("cases", [])] == list(range(10)),
                "COMPLETE_NATIVE_AGGREGATE_REQUIRED")
        for index, case in enumerate(cases["cases"]):
            row = aggregate["cases"][index]
            report = decoded[f"{label}_condition_{index:02d}"]
            binding = report.get("source_bindings", {})
            require(row.get("identity") == case["identity"] and row.get("native_job_id") == case["native_job_id"]
                    and row.get("status") == report.get("status") == NATIVE_PASS
                    and str(report.get("job_id")) == str(execution["legacy_compatibility_id"])
                    and report.get("units") == 230924 and report.get("duration_ms") == 3000.0
                    and report.get("runtime_code") == expected["runtime_code"]
                    and report.get("component_rtol") == report.get("component_atol") == 1e-12
                    and report.get("cuda_execution", {}).get("library_sha256") == CANDIDATE_LIBRARY_SHA256
                    and str(binding.get("native_job_id")) == str(case["native_job_id"])
                    and binding.get("files", {}).get("primary/report.json", {}).get("sha256")
                    == case["native_report_sha256"]
                    and all(report.get(key) is True and row.get(key) is True for key in
                            ("all_spike_pairs_exact", "all_unit_counts_exact",
                             "terminal_voltage_within_component_tolerance")),
                    "NATIVE_CONDITION_EVIDENCE_DIFFERS")
    replay = decoded["replay_comparison"]
    require(replay.get("planned_conditions") == replay.get("passed_conditions") == 10
            and replay.get("failed_conditions") == 0
            and [row.get("condition_index") for row in replay.get("conditions", [])] == list(range(10))
            and all(row.get("passed") is True and set(row.get("checks", {})) == REPLAY_CHECKS
                    and _all_true(row.get("checks"))
                    for row in replay["conditions"]), "COMPLETE_INDEPENDENT_REPLAY_REQUIRED")
    require(family.get("independent_replay") == replay, "REPLAY_SUMMARY_DIFFERS")
    return deepcopy(backend)


@lru_cache(maxsize=32)
def _validated_payload(payload: str, sha256: str, original_sha256: str, backend_json: str) -> str:
    require(digest_bytes(payload.encode()) == sha256, "ARTIFACT_HASH_DIFFERS")
    backend = validate_artifact(json.loads(payload), original_sha256, json.loads(backend_json))
    return json.dumps(backend, sort_keys=True)


def validate_document(document: dict, original_sha256: str, original_backend: dict) -> dict:
    require(isinstance(document, dict) and isinstance(document.get("payload"), str)
            and isinstance(document.get("sha256"), str), "AUTHENTICATED_ARTIFACT_REQUIRED")
    return json.loads(_validated_payload(document["payload"], document["sha256"], original_sha256,
                                        json.dumps(original_backend, sort_keys=True)))


def load(path: Path, original_sha256: str, original_backend: dict) -> tuple[dict, dict]:
    document = source_document(path)
    return document, validate_document(document, original_sha256, original_backend)



def registered_backend(document: dict, original_sha256: str) -> dict:
    artifact = decode_document(document)
    original = artifact.get("documents", {}).get("original_qualification", {})
    require(original.get("sha256") == original_sha256, "REGISTRY_ORIGINAL_BINDING_DIFFERS")
    proof = decode_document(original)
    return validate_document(document, original_sha256, proof["source_recipes"]["glif_cuda"])


def validate_registry(binding: dict, registry: dict) -> None:
    require(isinstance(registry, dict), "REGISTRY_REQUIRED")
    for sha, document in registry.items():
        require(binding.get("scenario_id") == "v1_visual"
                and document.get("sha256") == sha, "REGISTRY_IDENTITY_DIFFERS")
        registered_backend(document, binding["source_qualification_sha256"])


def validate_episode(binding: dict, registry: dict, metadata: dict, row: dict) -> None:
    admission = metadata.get("source_record", {}).get("production_source_admission", {})
    sha = admission.get("runtime_qualification_sha256")
    expected = row.get("runtime_qualification_sha256")
    require(sha == expected, "EPISODE_RUNTIME_REFERENCE_DIFFERS")
    if sha is None:
        # Preserve generic older qualified backends. This added candidate needs
        # an explicit proof even when both serialized references were removed.
        if admission.get("execution_backend") == "glif_cuda":
            for role in ("execution", "replay", "circuit_validation"):
                document = admission.get("source_documents", {}).get(role)
                if isinstance(document, dict) and isinstance(document.get("payload"), str):
                    report = decode_document(document)
                    require(report.get("cuda_execution", {}).get("library_sha256")
                            != CANDIDATE_LIBRARY_SHA256,
                            "KNOWN_CANDIDATE_REQUIRES_RUNTIME_IDENTITY")
        return
    require(binding.get("scope") == "production" and binding.get("scenario_id") == "v1_visual"
            and sha in registry, "EPISODE_RUNTIME_NOT_REGISTERED")
    backend = registered_backend(registry[sha], binding["source_qualification_sha256"])
    identity = admission.get("cohort_identity", {})
    require(admission.get("execution_backend") == "glif_cuda"
            and admission.get("source_qualification_sha256") == binding["source_qualification_sha256"]
            and admission.get("identity_table_sha256") == binding["identity_table_sha256"]
            and all(identity.get(key) == metadata.get(key)
                    for key in ("episode_id", "circuit_id", "block_id", "split"))
            and metadata.get("seeds") == {key: identity.get(key) for key in STREAMS},
            "EPISODE_COHORT_BINDING_DIFFERS")
    roles = {"execution": "execution_report_sha256", "replay": "replay_report_sha256",
             "circuit_validation": "circuit_validation_sha256"}
    for role, hash_key in roles.items():
        document = admission.get("source_documents", {}).get(role, {})
        require(document.get("sha256") == admission.get(hash_key), "EPISODE_REPORT_REFERENCE_DIFFERS")
        report = decode_document(document)
        require(report.get("runtime_code") == backend["runtime_code"]
                and report.get("cuda_execution", {}).get("library_sha256") == backend["library_sha256"],
                "EPISODE_RUNTIME_BACKEND_DIFFERS")
        reference = (report.get("source_bindings", {}) if role == "circuit_validation"
                     else report.get("circuit_bindings", {}))
        require(reference.get("files", {}).get("primary/report.json", {}).get("sha256")
                == admission.get("circuit_reference_report_sha256"), "EPISODE_CIRCUIT_REFERENCE_DIFFERS")


def _file_identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def prepare(original_qualification: Path, result_root: Path, receipt_path: Path,
            bundle: Path, return_manifest: Path, output: Path) -> dict:
    """Verify one returned bundle once and produce a self-contained admission proof."""
    receipt = json.loads(receipt_path.read_text())
    archive_identity = _file_identity(bundle)
    require(archive_identity == {"bytes": receipt["result_bytes"], "sha256": receipt["result_sha256"]},
            "RETURNED_ARCHIVE_DIFFERS")
    manifest_bytes = return_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        require(len(members) == len({member.name for member in members}),
                "RETURN_ARCHIVE_DUPLICATE_MEMBER")
        require(all(member.isfile() and not member.name.startswith("/")
                    and ".." not in Path(member.name).parts for member in members),
                "RETURN_ARCHIVE_MEMBER_DIFFERS")
        require({member.name for member in members} == set(manifest) | {"merlin_return_manifest.json"},
                "RETURN_ARCHIVE_INVENTORY_DIFFERS")
        require(archive.extractfile("merlin_return_manifest.json").read() == manifest_bytes,
                "RETURN_MANIFEST_NOT_IN_ARCHIVE")
    family = json.loads((result_root / "report.json").read_text())
    prefix = family["output"].lstrip("/") + "/"
    for name, expected in manifest.items():
        require(name.startswith(prefix), "RETURN_FILE_OUTSIDE_RESULT")
        path = result_root / name[len(prefix):]
        require(path.resolve().is_relative_to(result_root.resolve())
                and _file_identity(path) == expected, "RETURNED_FILE_DIFFERS:" + name)
    paths = {"compiler": "compiler_report.json", "cases": "frozen_cases.json",
             "snapshot": "snapshot_report.json", "family_report": "report.json",
             "primary": "primary/report.json", "independent_replay": "independent_replay/report.json",
             "replay_comparison": "independent_replay_comparison.json"}
    paths.update({f"{label}_condition_{i:02d}": f"{label}/condition_{i:02d}/report.json"
                  for label in ("primary", "independent_replay") for i in range(10)})
    documents = {role: source_document(result_root / path) for role, path in paths.items()}
    documents.update(original_qualification=source_document(original_qualification),
                     receipt=source_document(receipt_path), return_manifest=source_document(return_manifest))
    original = json.loads(original_qualification.read_text())["source_recipes"]["glif_cuda"]
    artifact = {
        "schema": SCHEMA, "status": "PASS",
        "original_qualification_sha256": documents["original_qualification"]["sha256"],
        "original_backend": {key: original[key] for key in
                             ("runtime_code", "library_sha256", "component_rtol", "component_atol")},
        "backend": dict(original, library_sha256=CANDIDATE_LIBRARY_SHA256, compute_capability=[8, 0]),
        "documents": documents, "evidence_paths": {role: prefix + path for role, path in paths.items()},
        "verified_return": {"archive_sha256": archive_identity["sha256"],
                            "archive_bytes": archive_identity["bytes"],
                            "manifest_sha256": documents["return_manifest"]["sha256"],
                            "verified_files": len(manifest)},
    }
    validate_artifact(artifact, artifact["original_qualification_sha256"], original)
    with output.open("x") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"status": "PASS_VERIFIED_GLIF_RUNTIME_QUALIFICATION", "path": str(output),
            **_file_identity(output)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-qualification", "result-root", "receipt", "bundle", "return-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID"),
            "MERLIN_VERIFIED_RETURN_PREPARATION_REQUIRED")
    print(json.dumps(prepare(args.original_qualification, args.result_root, args.receipt,
                             args.bundle, args.return_manifest, args.output)))


if __name__ == "__main__":
    main()


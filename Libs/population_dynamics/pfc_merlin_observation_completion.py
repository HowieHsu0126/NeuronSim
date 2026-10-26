"""Complete the frozen PFC observation manifest from existing immutable readouts.

Request schema pfc_merlin_observation_completion_v1 has runtime and accepted_1906
path/bytes/sha256 descriptors, preparation_sha256, collector_progress_uri, work,
summary_output, dynamics_output, summary_command, dynamics_command,
publication_roots and consumer_sha256 (the qualified overlay source hashes).
Bootstrap provides baseline tasks0..34 and the original prepared plan. No source
simulation or readout generation is performed. Observation restoration,
persistence and publication use the verified external runtime. Native sources
also restore four hash-bound source reports from the same verified bundle.

--prepare-supplements additionally requires accepted_sources (ten indexed local
bindings) and prior_completion_progress (a bound a5 progress wrapper). It only
prepares reusable snapshots. The default completion path remains unchanged.
A prepared_supplements completion also requires previous_execution, a bound
existing a5 operational record confirming failure and nonzero final exit code.
Bootstrap must explicitly stage accepted_sources, accepted_1906 and the prior
completion progress; producer paths in preparation reports are never followed.
source_readout_publications adds single-ID request/report bindings to this
handoff. It preserves the ten-source contract and previous-execution gate.

--publish-source-readout uses schema pfc_merlin_source_readout_publication_v1.
It requires index, accepted_source and scientific_intake local bindings,
source_execution, publication_execution, runtime, preparation_sha256,
consumer_sha256 and a fresh work directory. publication_kind defaults to
recovery, which retains the separate publisher contract. native_source requires
an original receipt published by the same real source run/trial. It publishes
one existing readout snapshot. It does not authorize a consumer or qualify the
scientific source.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tarfile
import tempfile
import time
import traceback

from . import pfc_parameter_domain_summary as summary
from . import provenance
from .pfc_parameter_domain_quality import validate_plan

SCHEMA = "pfc_merlin_observation_completion_v1"
SOURCE_READOUT_SCHEMA = "pfc_merlin_source_readout_publication_v1"
RUNTIME_SHA256 = "ac6a9d50920b3a0185c8ba82f5d03ff3448ba3ea5c327e0911fd85318bcc53e5"
SOURCE_RUN = "6de397acc866aadd"
SOURCE_TRIAL = "435906314"
SOURCE_BUNDLE_BYTES = 86010908
SOURCE_BUNDLE_SHA256 = "441012fb8a3c48447a39d2038652446fa9bff36ac828e6a97fa184299dcbc92b"
COLLECTOR_RUN = "a04e7a088bad3ec2"
COLLECTOR_TRIAL = "436671746"
PREVIOUS_COMPLETION_RUN = "a5e61c466d9c82dc"
PREVIOUS_COMPLETION_TRIAL = "438705196"
SUPPLEMENTAL_SOURCE_INDICES = {843, 917, 1057, 1177, 1218, 1263, 1451, 2030, 2083, 2687}
BASELINE = set(range(35))
ORIGINAL_IDS = set(range(3000)) - {1906}
QUALIFICATION_FLAGS = ("source_qualification_granted", "scientific_parameter_domain_qualified",
                       "scientific_claim_qualified", "scaling_qualified", "uncertainty_coverage_qualified")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_valid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bound_bytes(binding, base):
    require(isinstance(binding, dict) and isinstance(binding.get("path"), str)
            and type(binding.get("bytes")) is int and binding["bytes"] >= 0
            and hash_valid(binding.get("sha256")), "LOCAL_INPUT_BINDING_REQUIRED")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = base / path
    data = path.read_bytes()
    require(len(data) == binding["bytes"] and hashlib.sha256(data).hexdigest() == binding["sha256"],
            "LOCAL_INPUT_BINDING_CHANGED")
    return path.resolve(), data


def load_runtime(binding, base):
    path, _ = bound_bytes(binding, base)
    require(binding["sha256"] == RUNTIME_SHA256, "QUALIFIED_RUNTIME_SOURCE_REQUIRED")
    spec = importlib.util.spec_from_file_location("pfc_immutable_readout_runtime", path)
    require(spec is not None and spec.loader is not None, "EXTERNAL_RUNTIME_UNLOADABLE")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def exact_indices(values, allowed):
    require(isinstance(values, list) and all(type(index) is int and index in allowed for index in values),
            "OBSERVATION_INDICES_INVALID")
    require(len(values) == len(set(values)), "OBSERVATION_INDICES_DUPLICATE")
    return set(values)


def validate_collector(progress, uri, previous_snapshots):
    require(isinstance(progress, dict) and progress.get("run_id") == COLLECTOR_RUN
            and progress.get("trial_id") == COLLECTOR_TRIAL
            and type(progress.get("required")) is int and progress["required"] == 3000
            and type(progress.get("production_episodes")) is int and progress["production_episodes"] == 0
            and progress.get("status") == "WAITING_COMPLETE3000_VERIFIED_READOUTS",
            "ORIGINAL_COLLECTOR_IDENTITY_OR_SCOPE_CHANGED")
    collected = exact_indices(progress.get("collected_indices"), ORIGINAL_IDS)
    require(type(progress.get("collected")) is int and progress["collected"] == len(collected)
            and BASELINE <= collected, "ORIGINAL_COLLECTOR_COUNT_CHANGED")
    snapshots = progress.get("readout_snapshots")
    require(isinstance(snapshots, list) and snapshots[:len(previous_snapshots)] == previous_snapshots,
            "IMMUTABLE_SNAPSHOT_HISTORY_CHANGED")
    covered, uris = set(), set()
    prefix = uri[:-len("/progress.json")] + "/readout_snapshots/"
    for snapshot in snapshots:
        require(isinstance(snapshot, dict), "ORIGINAL_SNAPSHOT_REQUIRED")
        indices = exact_indices(snapshot.get("indices"), ORIGINAL_IDS - BASELINE)
        require(indices and not covered & indices, "ORIGINAL_SNAPSHOT_OVERLAP")
        name = f"readouts_{min(indices):04d}_{max(indices):04d}/result.tar.gz"
        require(snapshot.get("run_id") == COLLECTOR_RUN and snapshot.get("trial_id") == COLLECTOR_TRIAL
                and type(snapshot.get("source_simulations")) is int and snapshot["source_simulations"] == 0
                and type(snapshot.get("production_episodes")) is int and snapshot["production_episodes"] == 0
                and type(snapshot.get("bundle_bytes")) is int and snapshot["bundle_bytes"] > 0
                and hash_valid(snapshot.get("bundle_sha256")) and snapshot.get("bundle_uri") == prefix + name
                and snapshot["bundle_uri"] not in uris, "ORIGINAL_SNAPSHOT_SOURCE_OR_HASH_CHANGED")
        covered |= indices
        uris.add(snapshot["bundle_uri"])
    require(covered == collected - BASELINE, "ORIGINAL_SNAPSHOT_COVERAGE_CHANGED")
    return collected, snapshots


def observation_entry(runtime, index):
    directory = runtime.DOMAIN / "observations" / f"task_{index:04d}"
    return dict(path=str(directory), development_report_sha256=runtime.sha(directory / "development5ms/report.json"),
                observation_report_sha256=runtime.sha(directory / "observation1ms/report.json"))


def accepted_source_snapshot(accepted, expected, runtime):
    require(isinstance(accepted, dict) and accepted.get("status") == "ACCEPTED_MERLIN_DEVELOPMENT_EPISODE"
            and type(accepted.get("index")) is int and accepted["index"] == 1906
            and accepted.get("identity") == expected["identity"]
            and accepted.get("run_id") == SOURCE_RUN and accepted.get("trial_id") == SOURCE_TRIAL
            and type(accepted.get("source_simulations")) is int and accepted["source_simulations"] == 0
            and type(accepted.get("production_episodes")) is int and accepted["production_episodes"] == 0
            and all(accepted.get(field) is False for field in
                    ("source_qualification_granted", "scientific_parameter_domain_qualified", "scaling_qualified")),
            "ACCEPTED1906_SOURCE_IDENTITY_CHANGED")
    uri = accepted.get("source_bundle_uri")
    require(isinstance(uri, str) and uri.startswith("hdfs://")
            and uri.endswith("/" + SOURCE_RUN + "/pfc/task_1906/result.tar.gz")
            and accepted.get("source_bundle_sha256") == SOURCE_BUNDLE_SHA256,
            "ACCEPTED1906_ORIGINAL_BUNDLE_REQUIRED")
    entry = accepted.get("observation_entry")
    require(isinstance(entry, dict) and set(entry) == {"path", "development_report_sha256", "observation_report_sha256"}
            and entry["path"] == str(runtime.DOMAIN / "observations/task_1906")
            and hash_valid(entry["development_report_sha256"]) and hash_valid(entry["observation_report_sha256"]),
            "ACCEPTED1906_BOUND_OBSERVATION_REQUIRED")
    return dict(indices=[1906], bundle_uri=uri, bundle_bytes=SOURCE_BUNDLE_BYTES,
                bundle_sha256=SOURCE_BUNDLE_SHA256, run_id=SOURCE_RUN, trial_id=SOURCE_TRIAL,
                source_simulations=0, production_episodes=0)



def validate_authorized_stop(record, run_id, trial_id):
    """Bind an explicit safe-stop reply to the real stopped predecessor."""
    allowed = {"a5e61c466d9c82dc", "28bf3f86a34814e0"}
    require(isinstance(record, dict) and run_id in allowed
            and record.get("run_id") == run_id and record.get("trial_id") == trial_id
            and record.get("platform_status") == "stopped" and record.get("terminal_confirmed") is True
            and "final_exit_code" in record
            and (record["final_exit_code"] is None or type(record["final_exit_code"]) is int),
            "AUTHORIZED_PREDECESSOR_STOP_IDENTITY_REQUIRED")
    authorization = record.get("user_authorization")
    require(isinstance(authorization, dict) and authorization.get("source") == "user_explicit_reply"
            and authorization.get("date") == "2026-10-26"
            and isinstance(authorization.get("allowed_run_ids"), list)
            and len(authorization["allowed_run_ids"]) == 2
            and all(isinstance(value, str) for value in authorization["allowed_run_ids"])
            and set(authorization["allowed_run_ids"]) == allowed
            and isinstance(authorization.get("actions"), list) and len(authorization["actions"]) == 2
            and all(isinstance(value, str) for value in authorization["actions"])
            and set(authorization["actions"]) == {"safe_stop", "unfinished_takeover"},
            "EXPLICIT_PREDECESSOR_STOP_AUTHORIZATION_REQUIRED")
    operation = record.get("stop_operation")
    require(isinstance(operation, dict) and operation.get("run_id") == run_id
            and operation.get("trial_id") == trial_id and operation.get("action") == "stop"
            and operation.get("accepted") is True and type(operation.get("response_base_code")) is int
            and operation["response_base_code"] == 0, "ACCEPTED_PREDECESSOR_STOP_OPERATION_REQUIRED")
    terminal = record.get("terminal_observation")
    require(isinstance(terminal, dict) and terminal.get("run_id") == run_id and terminal.get("trial_id") == trial_id
            and terminal.get("platform_status") == "stopped" and terminal.get("terminal_confirmed") is True,
            "CONFIRMED_PREDECESSOR_STOP_OBSERVATION_REQUIRED")
    for evidence in (operation, terminal):
        value = evidence.get("observed_utc")
        require(isinstance(value, str) and value, "PREDECESSOR_STOP_OBSERVATION_TIME_REQUIRED")
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("PREDECESSOR_STOP_OBSERVATION_TIME_INVALID") from error
        require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0,
                "PREDECESSOR_STOP_OBSERVATION_TIME_INVALID")
    return dict(termination_kind="user_authorized_stop", user_authorization=authorization,
                stop_operation=operation, terminal_observation=terminal)


def validate_previous_execution(binding, base):
    path, data = bound_bytes(binding, base)
    previous = json.loads(data)
    require(isinstance(previous, dict) and previous.get("run_id") == PREVIOUS_COMPLETION_RUN
            and previous.get("trial_id") == PREVIOUS_COMPLETION_TRIAL,
            "PREVIOUS_COMPLETION_CONFIRMED_FAILURE_REQUIRED")
    stop_proof = {}
    if previous.get("platform_status") == "stopped":
        stop_proof = validate_authorized_stop(previous, PREVIOUS_COMPLETION_RUN, PREVIOUS_COMPLETION_TRIAL)
    else:
        require(previous.get("platform_status") == "failed" and previous.get("terminal_confirmed") is True
                and type(previous.get("final_exit_code")) is int and previous["final_exit_code"] != 0,
                "PREVIOUS_COMPLETION_CONFIRMED_FAILURE_REQUIRED")
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                run_id=previous["run_id"], trial_id=previous["trial_id"], platform_status=previous["platform_status"],
                terminal_confirmed=previous["terminal_confirmed"], final_exit_code=previous["final_exit_code"], **stop_proof)

def same_bound_input(actual, recorded, message):
    # Recorded producer paths are provenance, never local input instructions.
    require(isinstance(recorded, dict) and isinstance(recorded.get("path"), str)
            and type(recorded.get("bytes")) is int and hash_valid(recorded.get("sha256"))
            and recorded["bytes"] == actual["bytes"] and recorded["sha256"] == actual["sha256"], message)


def prepared_supplement_handoff(request, base, plan, runtime, accepted):
    path, data = bound_bytes(request.get("prepared_supplements"), base)
    prepared = json.loads(data)
    require(isinstance(prepared, dict) and prepared.get("schema") == SCHEMA
            and prepared.get("operation") == "prepare_supplements" and prepared.get("status") == "SUPPLEMENTAL_READOUTS_READY"
            and prepared.get("runtime_sha256") == RUNTIME_SHA256
            and prepared.get("preparation_sha256") == request["preparation_sha256"]
            and prepared.get("consumer_sha256") == request["consumer_sha256"]
            and prepared.get("failures") == [] and prepared.get("closure_verified") is False
            and type(prepared.get("required_episodes")) is int and prepared["required_episodes"] == 3000
            and type(prepared.get("prepared_episodes")) is int and prepared["prepared_episodes"] == 10
            and type(prepared.get("reused_persisted_episodes")) is int and prepared["reused_persisted_episodes"] == 1
            and all(type(prepared.get(field)) is int and prepared[field] == 0 for field in
                ("source_simulations", "production_episodes", "source_execution_reruns", "readout_execution_reruns",
                 "summary_executions", "dynamics_executions"))
            and all(prepared.get(field) is False for field in QUALIFICATION_FLAGS), "PREPARED_SUPPLEMENTS_CONTRACT_CHANGED")
    require(exact_indices(prepared.get("prepared_indices"), SUPPLEMENTAL_SOURCE_INDICES) == SUPPLEMENTAL_SOURCE_INDICES
            and exact_indices(prepared.get("available_recovery_indices"), SUPPLEMENTAL_SOURCE_INDICES | {1906})
                == SUPPLEMENTAL_SOURCE_INDICES | {1906}, "PREPARED_SUPPLEMENTS_COVERAGE_CHANGED")
    publisher = prepared.get("execution")
    require(isinstance(publisher, dict) and publisher.get("backend") == "merlin"
            and type(publisher.get("allocated_gpus")) is int and publisher["allocated_gpus"] == 0
            and isinstance(publisher.get("run_id"), str) and publisher["run_id"] != PREVIOUS_COMPLETION_RUN
            and bool(publisher["run_id"]) and isinstance(publisher.get("trial_id"), str) and publisher["trial_id"],
            "PREPARED_PUBLICATION_EXECUTION_CHANGED")
    prior, snapshot1906, owner = prior_1906_readout(request.get("prior_completion_progress"), base,
        request["accepted_1906"], accepted, plan, runtime, request)
    same_bound_input(prior, prepared.get("prior_completion_progress"), "PREPARED_PRIOR_COMPLETION_BINDING_CHANGED")
    require(prepared["prior_completion_progress"].get("uri") == prior["uri"], "PREPARED_PRIOR_COMPLETION_BINDING_CHANGED")
    same_bound_input(request["accepted_1906"], prepared.get("accepted_1906_binding"), "PREPARED1906_ACCEPTANCE_BINDING_CHANGED")
    require(prepared.get("consumer_owner_execution") == owner and prepared.get("reused_1906_snapshot") == snapshot1906
            and prepared.get("observation_1906_entry") == accepted["observation_entry"], "PREPARED1906_READOUT_BINDING_CHANGED")
    bindings = request.get("accepted_sources")
    require(isinstance(bindings, list) and len(bindings) == 10, "STAGED_SUPPLEMENTAL_ACCEPTANCES_REQUIRED")
    staged = [supplemental_source(binding, base, plan, runtime) for binding in bindings]
    require(exact_indices([source["index"] for source in staged], SUPPLEMENTAL_SOURCE_INDICES)
            == SUPPLEMENTAL_SOURCE_INDICES, "STAGED_SUPPLEMENTAL_ACCEPTANCES_REQUIRED")
    records = prepared.get("supplements")
    require(isinstance(records, list) and len(records) == 10, "PREPARED_SUPPLEMENTAL_RECORDS_CHANGED")
    require(exact_indices([record.get("index") for record in records], SUPPLEMENTAL_SOURCE_INDICES)
            == SUPPLEMENTAL_SOURCE_INDICES, "PREPARED_SUPPLEMENTAL_RECORDS_CHANGED")
    records = {record["index"]:record for record in records}
    sources = {}
    for source in staged:
        index = source["index"]
        record = records[index]
        same_bound_input(source["acceptance_binding"], record.get("acceptance_binding"), "PREPARED_SOURCE_ACCEPTANCE_BINDING_CHANGED")
        require(all(record.get(key) == value for key,value in source.items() if key != "acceptance_binding"),
                "PREPARED_SOURCE_BINDING_CHANGED")
        source["readout_snapshot"] = validate_single_readout_snapshot(record.get("readout_snapshot"), index, publisher)
        sources[index] = source
    return dict(binding=dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
                prior_completion_progress=prior, sources=sources, reused_1906_snapshot=snapshot1906,
                publication_execution=publisher)

def published_source_handoff(request, base, plan, runtime):
    bindings = request.get("source_readout_publications", [])
    require(isinstance(bindings, list), "SOURCE_READOUT_PUBLICATION_BINDINGS_REQUIRED")
    sources = {}
    for binding in bindings:
        require(isinstance(binding, dict), "SOURCE_READOUT_PUBLICATION_BINDINGS_REQUIRED")
        path, data = bound_bytes(binding.get("source_readout_request"), base)
        single = json.loads(data)
        require(isinstance(single, dict) and single.get("schema") == SOURCE_READOUT_SCHEMA
                and single.get("preparation_sha256") == request["preparation_sha256"]
                and single.get("consumer_sha256") == request["consumer_sha256"],
                "SOURCE_READOUT_PUBLICATION_PLAN_CHANGED")
        same_bound_input(request["runtime"], single.get("runtime"), "SOURCE_READOUT_PUBLICATION_RUNTIME_CHANGED")
        source, intake_binding, publisher = single_source_publication_inputs(single, path.parent, plan, runtime)
        index = source["index"]
        require(index not in sources and index not in SUPPLEMENTAL_SOURCE_INDICES | BASELINE | {1906},
                "SOURCE_READOUT_PUBLICATION_DUPLICATE_OR_RESERVED_INDEX")
        report_path, report_data = bound_bytes(binding.get("source_readout_publication"), base)
        published = json.loads(report_data)
        require(isinstance(published, dict) and published.get("schema") == SOURCE_READOUT_SCHEMA
                and published.get("operation") == "publish_source_readout"
                and published.get("status") == "SOURCE_READOUT_READY"
                and published.get("runtime_sha256") == RUNTIME_SHA256
                and published.get("preparation_sha256") == request["preparation_sha256"]
                and published.get("consumer_sha256") == request["consumer_sha256"]
                and published.get("publication_kind") == single.get("publication_kind", "recovery")
                and published.get("publication_execution") == publisher
                and type(published.get("requested_index")) is int and published["requested_index"] == index
                and published.get("published_indices") == [index]
                and type(published.get("published_episodes")) is int and published["published_episodes"] == 1
                and type(published.get("required_episodes")) is int and published["required_episodes"] == 3000
                and published.get("failures") == [] and published.get("closure_verified") is False
                and published.get("consumer_start_authorized") is False
                and all(type(published.get(field)) is int and published[field] == 0 for field in
                    ("source_simulations", "production_episodes", "source_execution_reruns", "readout_execution_reruns",
                     "summary_executions", "dynamics_executions"))
                and all(published.get(field) is False for field in QUALIFICATION_FLAGS),
                "SOURCE_READOUT_PUBLICATION_CONTRACT_CHANGED")
        execution = published.get("execution")
        require(isinstance(execution, dict) and execution.get("backend") == "merlin"
                and type(execution.get("allocated_gpus")) is int and execution["allocated_gpus"] == 0
                and isinstance(execution.get("run_id"), str) and execution["run_id"]
                and execution["run_id"] not in {source["source_execution"]["run_id"], publisher["run_id"], PREVIOUS_COMPLETION_RUN}
                and isinstance(execution.get("trial_id"), str) and execution["trial_id"],
                "SOURCE_READOUT_PUBLICATION_EXECUTION_CHANGED")
        recorded = published.get("source")
        require(isinstance(recorded, dict) and set(recorded) == set(source)
                and all(recorded.get(key) == value for key, value in source.items() if key != "acceptance_binding"),
                "SOURCE_READOUT_PUBLICATION_SOURCE_CHANGED")
        same_bound_input(source["acceptance_binding"], recorded.get("acceptance_binding"),
                         "SOURCE_READOUT_PUBLICATION_ACCEPTANCE_CHANGED")
        same_bound_input(intake_binding, published.get("scientific_intake_binding"),
                         "SOURCE_READOUT_PUBLICATION_INTAKE_CHANGED")
        sources[index] = dict(source,
            readout_snapshot=validate_single_readout_snapshot(published.get("readout_snapshot"), index, execution),
            publication_kind=published["publication_kind"],
            source_request_binding=dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
            publication_binding=dict(path=str(report_path), bytes=len(report_data), sha256=hashlib.sha256(report_data).hexdigest()),
            publication_execution=execution)
    return sources


def run(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_bytes())
    require(isinstance(request, dict) and request.get("schema") == SCHEMA, "COMPLETION_REQUEST_SCHEMA_INVALID")
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0", "MERLIN_CPU_COMPLETION_REQUIRED")
    supplemental_mode = any(key in request for key in ("prepared_supplements", "previous_execution", "source_readout_publications"))
    previous_execution = validate_previous_execution(request.get("previous_execution"), request_path.parent) if supplemental_mode else None
    runtime = load_runtime(request.get("runtime"), request_path.parent)
    work = Path(request["work"]).resolve()
    summary_output, dynamics_output = Path(request["summary_output"]).resolve(), Path(request["dynamics_output"]).resolve()
    require(not work.exists() and not summary_output.exists() and not dynamics_output.exists()
            and len({work, summary_output, dynamics_output}) == 3, "FRESH_COMPLETION_OUTPUTS_REQUIRED")
    work.mkdir(parents=True)
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                     hostname=socket.gethostname(), allocated_gpus=0)
    state = dict(schema=SCHEMA, status="RESTORING_ACCEPTED1906_READOUT", execution=execution,
        collector_execution=dict(run_id=COLLECTOR_RUN, trial_id=COLLECTOR_TRIAL), required_episodes=3000,
        runtime_sha256=RUNTIME_SHA256, source_simulations=0, source_execution_reruns=0, readout_execution_reruns=0,
        production_episodes=0, independent_histories_per_circuit=3, histories_are_descriptive_only=True,
        failures=[], **{field:False for field in QUALIFICATION_FLAGS})
    manifest_path, report_path, log_path = work / "complete_observations.json", work / "completion_report.json", work / "consumer.log"
    done, previous_snapshots = set(), []
    roots = [Path(path).resolve() for path in request["publication_roots"]]
    require(summary_output in roots and dynamics_output in roots, "COMPLETE_CONSUMER_PUBLICATION_ROOTS_REQUIRED")
    roots += [manifest_path, report_path, log_path]

    def report():
        state.update(collected=len(done), collected_indices=sorted(done))
        report_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        runtime.publish_json("observation_completion_progress.json", state)

    try:
        hashes = request.get("consumer_sha256")
        require(isinstance(hashes, dict)
                and hashes.get("pfc_parameter_domain_summary.py") == sha(summary.__file__)
                and hashes.get("provenance.py") == sha(provenance.__file__)
                and hash_valid(hashes.get("pfc_parameter_task_dynamics.py")), "QUALIFIED_CONSUMER_OVERLAY_REQUIRED")
        state["consumer_sha256"] = hashes
        prepared = runtime.DOMAIN / "prepared"
        require(hash_valid(request.get("preparation_sha256"))
                and sha(prepared / "execution.json") == request["preparation_sha256"], "FROZEN_PREPARATION_CHANGED")
        plan = json.loads((prepared / "execution.json").read_bytes())
        validate_plan(plan)
        require(isinstance(plan.get("rows"), list) and len(plan["rows"]) == 3000
                and [row.get("task_index") for row in plan["rows"]] == list(range(3000)), "COMPLETE3000_PLAN_REQUIRED")
        episode_ids = [row["identity"]["episode_id"] for row in plan["rows"]]
        require(len(set(episode_ids)) == 3000, "DUPLICATE_PLAN_EPISODE")
        state["preparation_sha256"] = request["preparation_sha256"]
        expected_summary = [request["summary_command"][0], "-B", "-m", "population_dynamics.pfc_parameter_domain_summary",
            "--prepared", str(prepared), "--observations", str(manifest_path), "--output", str(summary_output)]
        require(request["summary_command"] == expected_summary
                and isinstance(request.get("dynamics_command"), list) and request["dynamics_command"]
                and all(isinstance(value, str) and value for value in request["dynamics_command"]),
                "UNCHANGED_COMPLETE_CONSUMER_COMMANDS_REQUIRED")
        collector_uri = request.get("collector_progress_uri")
        require(isinstance(collector_uri, str) and collector_uri.startswith("hdfs://")
                and collector_uri.endswith("/" + COLLECTOR_RUN + "/pfc_downstream/progress.json"),
                "ORIGINAL_COLLECTOR_PROGRESS_URI_REQUIRED")
        # Baseline files are hash-bound in the staged input archive. One real
        # baseline validates this environment; aggregate validates all3000.
        summary.load_observation(observation_entry(runtime, 0), plan["rows"][0])
        done.update(BASELINE)
        accepted_path, accepted_data = bound_bytes(request.get("accepted_1906"), request_path.parent)
        accepted = json.loads(accepted_data)
        source_snapshot = accepted_source_snapshot(accepted, plan["rows"][1906], runtime)
        state.update(accepted_1906_metadata_path=str(accepted_path), accepted_1906_metadata_sha256=hashlib.sha256(accepted_data).hexdigest(),
                     source_1906_snapshot=source_snapshot)
        handoff = None
        published_sources = {}
        if supplemental_mode:
            handoff = prepared_supplement_handoff(request, request_path.parent, plan, runtime, accepted)
            published_sources = published_source_handoff(request, request_path.parent, plan, runtime)
            state.update(previous_execution=previous_execution, prepared_supplements_binding=handoff["binding"],
                         supplemental_publication_execution=handoff["publication_execution"],
                         reused_1906_snapshot=handoff["reused_1906_snapshot"], status="WAITING_BOUND_SUPPLEMENTAL_UNION")
            if published_sources:
                state["source_readout_publications"] = {str(index):source for index, source in published_sources.items()}
        else:
            # restore_snapshot delegates download, size/SHA checks and extraction to
            # the qualified runtime's unpack_readouts. The source bundle is unchanged.
            require(runtime.restore_snapshot(source_snapshot) == [1906], "RESTORED1906_IDENTITY_CHANGED")
            entry = observation_entry(runtime, 1906)
            require(entry == accepted["observation_entry"], "RESTORED1906_ACCEPTED_REPORT_HASH_CHANGED")
            summary.load_observation(entry, plan["rows"][1906])
            recovered_snapshot = runtime.persist_readouts([1906], plan, lambda index: observation_entry(runtime, index))
            require(recovered_snapshot.get("indices") == [1906]
                    and recovered_snapshot.get("run_id") == execution["run_id"]
                    and recovered_snapshot.get("trial_id") == execution["trial_id"]
                    and recovered_snapshot.get("source_simulations") == recovered_snapshot.get("production_episodes") == 0,
                    "RECOVERED1906_SNAPSHOT_PUBLICATION_CHANGED")
            done.add(1906)
            state.update(recovered_1906_snapshot=recovered_snapshot, status="WAITING_ORIGINAL2999_SNAPSHOTS")
        report()
        while True:
            progress = runtime.hdfs_json(collector_uri)
            require(progress is not None, "ORIGINAL_COLLECTOR_PROGRESS_MISSING")
            collected, snapshots = validate_collector(progress, collector_uri, previous_snapshots)
            previous_snapshots = snapshots
            state.update(observed_collector_count=len(collected), observed_collector_snapshots=len(snapshots))
            report()
            if handoff is not None:
                available = collected | SUPPLEMENTAL_SOURCE_INDICES | {1906} | set(published_sources)
                state.update(available_indices=sorted(available), available_count=len(available))
                if available == set(range(3000)):
                    state.update(fixed_original_collected_indices=sorted(collected), fixed_original_snapshots=snapshots,
                                 original_collector_progress_uri=collector_uri)
                    break
            elif collected == ORIGINAL_IDS:
                break
            time.sleep(180)
        state["status"] = "RESTORING_FIXED_ORIGINAL_SNAPSHOTS" if handoff is not None else "RESTORING_ORIGINAL2999_SNAPSHOTS"
        checked_overlaps = set()
        report()
        for number, snapshot in enumerate(snapshots, 1):
            restored = runtime.restore_snapshot(snapshot)
            require(restored == snapshot["indices"] and not set(restored) & done, "RESTORED_SNAPSHOT_IDENTITY_OR_OVERLAP_CHANGED")
            # restore_snapshot authenticates every archived file. The complete
            # summary performs the existing numerical checks once per episode.
            if handoff is not None:
                for index in sorted(set(restored) & SUPPLEMENTAL_SOURCE_INDICES):
                    require(observation_entry(runtime, index) == handoff["sources"][index]["observation_entry"],
                            "ORIGINAL_SUPPLEMENTAL_REPORT_CONFLICT:" + str(index))
                    checked_overlaps.add(index)
                for index in sorted(set(restored) & set(published_sources)):
                    require(observation_entry(runtime, index) == published_sources[index]["observation_entry"],
                            "ORIGINAL_PUBLISHED_SOURCE_REPORT_CONFLICT:" + str(index))
                    checked_overlaps.add(index)
            done.update(restored)
            state["restored_original_snapshots"] = number
            if number % 50 == 0:
                report()
        if handoff is not None:
            restored_supplements = []
            for index in sorted(SUPPLEMENTAL_SOURCE_INDICES | {1906}):
                if index in done:
                    require(index in checked_overlaps, "UNVERIFIED_SUPPLEMENTAL_OVERLAP")
                    continue
                snapshot = handoff["reused_1906_snapshot"] if index == 1906 else handoff["sources"][index]["readout_snapshot"]
                expected_entry = accepted["observation_entry"] if index == 1906 else handoff["sources"][index]["observation_entry"]
                require(runtime.restore_snapshot(snapshot) == [index], "PREPARED_RESTORED_IDENTITY_CHANGED")
                require(observation_entry(runtime, index) == expected_entry, "PREPARED_RESTORED_REPORT_CONFLICT:" + str(index))
                done.add(index)
                restored_supplements.append(index)
            state.update(source_overlap_indices=sorted(checked_overlaps), restored_prepared_indices=restored_supplements)
            restored_publications = []
            for index, source in sorted(published_sources.items()):
                if index in done:
                    require(index in checked_overlaps, "UNVERIFIED_PUBLISHED_SOURCE_OVERLAP")
                    continue
                restored = (restore_native_source_readout(source, runtime) if source["publication_kind"] == "native_source"
                            else runtime.restore_snapshot(source["readout_snapshot"]))
                require(restored == [index], "PUBLISHED_SOURCE_RESTORED_IDENTITY_CHANGED")
                require(observation_entry(runtime, index) == source["observation_entry"],
                        "PUBLISHED_SOURCE_RESTORED_REPORT_CONFLICT:" + str(index))
                done.add(index)
                restored_publications.append(index)
            if published_sources:
                state["restored_source_readout_publication_indices"] = restored_publications
        require(done == set(range(3000)), "EXACT3000_OBSERVATION_COVERAGE_REQUIRED")
        manifest = {row["identity"]["episode_id"]:observation_entry(runtime, row["task_index"]) for row in plan["rows"]}
        provenance.freeze_json(manifest_path, manifest)
        state.update(status="COMPLETE3000_OBSERVATIONS_RUNNING_CONSUMERS", observation_manifest_sha256=sha(manifest_path))
        report()
        state["consumer_commands"] = [request["summary_command"], request["dynamics_command"]]
        with log_path.open("x", encoding="utf-8") as log:
            for argv in state["consumer_commands"]:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
                state["consumer_exit_code"] = result.returncode
                require(result.returncode == 0, "COMPLETE_CONSUMER_FAILED:exit" + str(result.returncode))
        state.update(status="COMPLETE3000_DESCRIPTIVE_OBSERVATION_HANDOFF", consumer_exit_code=0)
        report()
        runtime.publish_results(roots, 0)
        return state
    except Exception as error:
        state.update(status="FAILED_OBSERVATION_COMPLETION", failures=[dict(error_type=type(error).__name__,
                     message=str(error), traceback=traceback.format_exc())])
        report_path.write_text(json.dumps(dict(state, collected=len(done), collected_indices=sorted(done)),
                                         indent=2, sort_keys=True, allow_nan=False) + "\n")
        try:
            runtime.publish_json("observation_completion_failure.json", state)
            runtime.publish_results(roots, state.get("consumer_exit_code") or 1)
        except Exception as publication_error:
            print("COMPLETION_FAILURE_PUBLICATION_FAILED:" + str(publication_error), flush=True)
        raise



def validate_single_readout_snapshot(snapshot, index, execution):
    require(isinstance(snapshot, dict) and snapshot.get("indices") == [index]
            and snapshot.get("run_id") == execution["run_id"]
            and snapshot.get("trial_id") == execution["trial_id"]
            and type(snapshot.get("bundle_bytes")) is int and snapshot["bundle_bytes"] > 0
            and hash_valid(snapshot.get("bundle_sha256"))
            and isinstance(snapshot.get("bundle_uri"), str)
            and snapshot["bundle_uri"].startswith("hdfs://")
            and snapshot["bundle_uri"].endswith("/" + execution["run_id"]
                + f"/pfc_downstream/readout_snapshots/readouts_{index:04d}_{index:04d}/result.tar.gz")
            and type(snapshot.get("source_simulations")) is int and snapshot["source_simulations"] == 0
            and type(snapshot.get("production_episodes")) is int and snapshot["production_episodes"] == 0,
            "SINGLE_READOUT_SNAPSHOT_BINDING_CHANGED")
    return snapshot


def supplemental_source(binding, base, plan, runtime):
    require(isinstance(binding, dict) and type(binding.get("index")) is int
            and binding["index"] in SUPPLEMENTAL_SOURCE_INDICES, "SUPPLEMENTAL_SOURCE_INDEX_INVALID")
    _, source = validated_readout_source(binding, base, plan, runtime)
    return source


def validated_readout_source(binding, base, plan, runtime, source_execution=None):
    # Legacy supplements retain their original source identity. Only the single
    # publication path supplies a separately bound real source execution.
    if source_execution is None:
        expected = dict(run_id=SOURCE_RUN, trial_id=SOURCE_TRIAL)
    else:
        require(isinstance(source_execution, dict) and set(source_execution) == {"run_id", "trial_id", "xie_job"}
                and all(isinstance(source_execution.get(key), str) and source_execution[key]
                        for key in ("run_id", "trial_id"))
                and type(source_execution.get("xie_job")) in (str, int) and str(source_execution["xie_job"]),
                "SINGLE_SOURCE_EXECUTION_REQUIRED")
        expected = source_execution
    source_run, source_trial = expected["run_id"], expected["trial_id"]
    require(isinstance(binding, dict) and type(binding.get("index")) is int
            and 0 <= binding["index"] < 3000, "SOURCE_READOUT_INDEX_INVALID")
    path, data = bound_bytes(binding, base)
    accepted = json.loads(data)
    index = binding["index"]
    require(isinstance(accepted, dict) and accepted.get("status") == "ACCEPTED_MERLIN_DEVELOPMENT_EPISODE"
            and type(accepted.get("index")) is int and accepted["index"] == index
            and accepted.get("identity") == plan["rows"][index]["identity"]
            and accepted.get("run_id") == source_run and accepted.get("trial_id") == source_trial
            and type(accepted.get("source_simulations")) is int and accepted["source_simulations"] == 0
            and type(accepted.get("production_episodes")) is int and accepted["production_episodes"] == 0
            and all(accepted.get(field) is False for field in
                ("source_qualification_granted", "scientific_parameter_domain_qualified", "scaling_qualified")),
            "SUPPLEMENTAL_SOURCE_ACCEPTANCE_CHANGED")
    receipt = accepted.get("remote_receipt")
    require(isinstance(receipt, dict) and receipt.get("status") == "COMPLETE"
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and receipt.get("persisted") is True and type(receipt.get("index")) is int
            and receipt["index"] == index and receipt.get("run_id") == source_run
            and receipt.get("trial_id") == source_trial
            and type(receipt.get("xie_job")) in (str, int) and str(receipt["xie_job"])
            and accepted.get("xie_job") == receipt["xie_job"], "SUPPLEMENTAL_SOURCE_RECEIPT_CHANGED")
    if "xie_job" in expected:
        require(receipt["xie_job"] == expected["xie_job"], "SINGLE_SOURCE_EXECUTION_CHANGED")
    recovery = receipt.get("publication_recovery", {})
    checkpoint = receipt.get("checkpoint_path")
    require(isinstance(recovery, dict) and isinstance(checkpoint, str), "SUPPLEMENTAL_SOURCE_RECEIPT_CHANGED")
    uri = recovery.get("source_bundle_uri", checkpoint.rstrip("/") + "/result.tar.gz")
    require(isinstance(uri, str) and uri.startswith("hdfs://")
            and uri.endswith("/" + source_run + f"/pfc/task_{index:04d}/result.tar.gz")
            and accepted.get("source_bundle_uri") == uri
            and hash_valid(receipt.get("result_sha256"))
            and accepted.get("source_bundle_sha256") == receipt["result_sha256"]
            and type(receipt.get("result_bytes")) is int and receipt["result_bytes"] > 0,
            "SUPPLEMENTAL_ORIGINAL_BUNDLE_CHANGED")
    validation = accepted.get("validation_execution")
    require(isinstance(validation, dict) and validation.get("backend") == "merlin"
            and validation.get("allocated_gpus") == 0 and validation.get("numpy_version") == "2.4.2"
            and isinstance(validation.get("run_id"), str) and validation["run_id"]
            and isinstance(validation.get("trial_id"), str) and validation["trial_id"],
            "SUPPLEMENTAL_VALIDATION_EXECUTION_CHANGED")
    entry = accepted.get("observation_entry")
    require(isinstance(entry, dict) and set(entry) == {"path", "development_report_sha256", "observation_report_sha256"}
            and entry["path"] == str(runtime.DOMAIN / "observations" / f"task_{index:04d}")
            and hash_valid(entry["development_report_sha256"]) and hash_valid(entry["observation_report_sha256"]),
            "SUPPLEMENTAL_OBSERVATION_BINDING_CHANGED")
    require(isinstance(accepted.get("receipt_uri"), str) and accepted["receipt_uri"].startswith("hdfs://"),
            "SUPPLEMENTAL_SOURCE_RECEIPT_URI_REQUIRED")
    snapshot = dict(indices=[index], bundle_uri=uri, bundle_bytes=receipt["result_bytes"],
                  bundle_sha256=receipt["result_sha256"], run_id=source_run, trial_id=source_trial,
                  source_simulations=0, production_episodes=0)
    source = dict(index=index, identity=accepted["identity"],
                acceptance_binding=dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
                source_execution=dict(run_id=source_run, trial_id=source_trial, xie_job=receipt["xie_job"]),
                validation_execution=validation, source_receipt_uri=accepted["receipt_uri"],
                 publication_recovery=recovery, source_bundle_snapshot=snapshot, observation_entry=entry)
    return accepted, source


def qualified_readout_plan(request, runtime):
    hashes = request.get("consumer_sha256")
    require(isinstance(hashes, dict)
            and hashes.get("pfc_parameter_domain_summary.py") == sha(summary.__file__)
            and hashes.get("provenance.py") == sha(provenance.__file__)
            and hash_valid(hashes.get("pfc_parameter_task_dynamics.py")), "QUALIFIED_CONSUMER_OVERLAY_REQUIRED")
    prepared = runtime.DOMAIN / "prepared"
    require(hash_valid(request.get("preparation_sha256"))
            and sha(prepared / "execution.json") == request["preparation_sha256"], "FROZEN_PREPARATION_CHANGED")
    plan = json.loads((prepared / "execution.json").read_bytes())
    validate_plan(plan)
    require(isinstance(plan.get("rows"), list) and len(plan["rows"]) == 3000
            and [row.get("task_index") for row in plan["rows"]] == list(range(3000)), "COMPLETE3000_PLAN_REQUIRED")
    require(len({row["identity"]["episode_id"] for row in plan["rows"]}) == 3000, "DUPLICATE_PLAN_EPISODE")
    return plan, hashes


def restore_native_source_readout(source, runtime):
    """Restore observations and the four source reports omitted by the legacy selector."""
    index = source["index"]
    _, accepted_data = bound_bytes(source["acceptance_binding"], runtime.DOMAIN)
    accepted = json.loads(accepted_data)
    source_archive = Path(accepted["source_archive"])
    require(source_archive.is_absolute() and source_archive.is_relative_to(runtime.DOMAIN)
            and source_archive.resolve() == source_archive, "NATIVE_SOURCE_ARCHIVE_PATH_CHANGED")
    snapshot = source.get("readout_snapshot", source["source_bundle_snapshot"])
    with tempfile.TemporaryDirectory(prefix=f"native_readout_{index:04d}_") as directory:
        bundle = Path(directory) / "result.tar.gz"
        subprocess.run(["hdfs", "dfs", "-get", snapshot["bundle_uri"], str(bundle)],
                       check=True, capture_output=True, text=True)
        require(bundle.stat().st_size == snapshot["bundle_bytes"] and sha(bundle) == snapshot["bundle_sha256"],
                "NATIVE_SOURCE_BUNDLE_CHANGED")
        runtime.unpack_readouts(bundle, snapshot["bundle_sha256"])
        development = json.loads((runtime.DOMAIN / "observations" / f"task_{index:04d}"
                                  / "development5ms/report.json").read_bytes())
        require(development.get("source_archive") == str(source_archive), "NATIVE_SOURCE_REPORT_ARCHIVE_CHANGED")
        required = [source_archive / name for name in (
            "delivery.json", "source_run/report.json", "qualification/delivery.json",
            "qualification/comparison/qualification.json")]
        with tarfile.open(bundle) as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            require(len(names) == len(set(names)) and all(member.isfile()
                    and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                    for member in members), "NATIVE_SOURCE_ARCHIVE_MEMBERS_CHANGED")
            manifest = json.load(archive.extractfile("merlin_return_manifest.json"))
            require(set(names) == set(manifest) | {"merlin_return_manifest.json"},
                    "NATIVE_SOURCE_ARCHIVE_MANIFEST_CHANGED")
            for target in required:
                name = str(target).lstrip("/")
                require(name in manifest and target.resolve() == target, "NATIVE_SOURCE_METADATA_MISSING_OR_UNSAFE")
                row = manifest[name]
                data = archive.extractfile(name).read()
                require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                        "NATIVE_SOURCE_METADATA_CHANGED")
                if target.exists():
                    require(target.read_bytes() == data, "NATIVE_SOURCE_EXISTING_METADATA_CHANGED")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    return [index]


def persist_source_readout(source, plan, runtime, execution, native=False):
    index = source["index"]
    restored = restore_native_source_readout(source, runtime) if native else runtime.restore_snapshot(source["source_bundle_snapshot"])
    require(restored == [index],
            "SUPPLEMENTAL_RESTORED_IDENTITY_CHANGED")
    require(observation_entry(runtime, index) == source["observation_entry"], "SUPPLEMENTAL_RESTORED_REPORT_CHANGED")
    snapshot = runtime.persist_readouts([index], plan, lambda task:observation_entry(runtime, task))
    return validate_single_readout_snapshot(snapshot, index, execution)


def single_source_publication_inputs(request, base, plan, runtime):
    index = request.get("index")
    binding = request.get("accepted_source")
    require(type(index) is int and 0 <= index < 3000 and isinstance(binding, dict)
            and type(binding.get("index")) is int and binding["index"] == index,
            "SINGLE_SOURCE_ACCEPTANCE_INDEX_CHANGED")
    source_execution = request.get("source_execution")
    require(source_execution is not None, "SINGLE_SOURCE_EXECUTION_REQUIRED")
    accepted, source = validated_readout_source(binding, base, plan, runtime, source_execution)
    require(source_execution == source["source_execution"], "SINGLE_SOURCE_EXECUTION_CHANGED")
    require(hash_valid(plan["rows"][index].get("config_sha256"))
            and accepted.get("config_sha256") == plan["rows"][index]["config_sha256"],
            "SINGLE_SOURCE_CONFIG_CHANGED")
    publisher = request.get("publication_execution")
    require(isinstance(publisher, dict) and set(publisher) == {"run_id", "trial_id"}
            and all(isinstance(publisher[key], str) and publisher[key] for key in publisher),
            "SOURCE_PUBLISHER_EXECUTION_REQUIRED")
    publication_kind = request.get("publication_kind", "recovery")
    require(publication_kind in ("native_source", "recovery"), "SOURCE_PUBLICATION_KIND_INVALID")
    receipt = accepted["remote_receipt"]
    if publication_kind == "native_source":
        # A native source runtime publishes its own original receipt and bundle.
        # A recovery receipt cannot be relabelled as native publication.
        require("publication_recovery" not in receipt
                and publisher == {key:source_execution[key] for key in ("run_id", "trial_id")},
                "NATIVE_SOURCE_PUBLISHER_CHANGED")
        checkpoint = receipt["checkpoint_path"].rstrip("/")
        require(checkpoint.startswith("hdfs://")
                and checkpoint.endswith("/" + source_execution["run_id"] + f"/pfc/task_{index:04d}")
                and source["source_receipt_uri"] == checkpoint + "/receipt.json"
                and source["source_bundle_snapshot"]["bundle_uri"] == checkpoint + "/result.tar.gz",
                "NATIVE_SOURCE_PUBLICATION_PATH_CHANGED")
    else:
        recovery = source["publication_recovery"]
        prefix = recovery.get("checkpoint_path")
        require(publisher["run_id"] != source_execution["run_id"]
                and all(recovery.get(key) == value for key,value in publisher.items())
                and isinstance(prefix, str) and prefix.startswith("hdfs://")
                and prefix.endswith("/" + publisher["run_id"])
                and type(recovery.get("allocated_gpus")) is int and recovery["allocated_gpus"] == 0
                and type(recovery.get("source_simulations")) is int and recovery["source_simulations"] == 0
                and recovery.get("scientific_admission_granted") is False, "SOURCE_PUBLISHER_EXECUTION_CHANGED")
        checkpoint = prefix + f"/pfc_publication/task_{index:04d}"
        require(receipt.get("checkpoint_path") == checkpoint
                and source["source_receipt_uri"] == checkpoint + "/receipt.json"
                and recovery.get("published_bundle_uri") == checkpoint + "/result.tar.gz"
                and recovery.get("source_checkpoint_path") == source["source_bundle_snapshot"]["bundle_uri"][:-len("/result.tar.gz")],
                "SOURCE_PUBLISHER_PATH_CHANGED")
    path, data = bound_bytes(request.get("scientific_intake"), base)
    intake = json.loads(data)
    require(isinstance(intake, dict) and intake.get("schema") == "pfc_merlin_scientific_intake_v1"
            and intake.get("status") == "PASS" and intake.get("failures") == []
            and all(type(intake.get(field)) is int and intake[field] == 0 for field in
                    ("source_simulations", "production_episodes"))
            and all(intake.get(field) is False for field in
                    ("source_qualification_granted", "scientific_parameter_domain_qualified", "scaling_qualified")),
            "PASS_SCIENTIFIC_INTAKE_REQUIRED")
    requested = exact_indices(intake.get("requested_task_indices"), set(range(3000)))
    completed = intake.get("completed")
    require(isinstance(completed, list) and all(isinstance(row, dict) for row in completed)
            and exact_indices([row.get("index") for row in completed], set(range(3000))) == requested
            and type(intake.get("new_accepted_episode_count")) is int
            and intake["new_accepted_episode_count"] == len(completed) and index in requested,
            "SCIENTIFIC_INTAKE_TASK_BINDING_CHANGED")
    record = next(row for row in completed if row["index"] == index)
    intake_checkpoint = intake.get("checkpoint_path")
    require(intake.get("execution") == accepted["validation_execution"]
            and isinstance(intake_checkpoint, str) and intake_checkpoint.startswith("hdfs://")
            and intake_checkpoint.endswith("/" + accepted["validation_execution"]["run_id"] + "/pfc_scientific_intake")
            and record.get("acceptance_uri") == intake_checkpoint + f"/task_{index:04d}_acceptance.json"
            and record.get("acceptance_sha256") == binding["sha256"], "SCIENTIFIC_INTAKE_ACCEPTANCE_CHANGED")
    return source, dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()), publisher


def publish_source_readout(request_path):
    """Publish one accepted existing readout without taking over a consumer."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_bytes())
    require(isinstance(request, dict) and request.get("schema") == SOURCE_READOUT_SCHEMA,
            "SOURCE_READOUT_REQUEST_SCHEMA_INVALID")
    run_id, trial_id = os.environ.get("RH2_JOB_RUN_ID"), os.environ.get("ARNOLD_TRIAL_ID")
    require(run_id and trial_id and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0"
            and os.environ.get("ARNOLD_MONITOR_ROLE") == "head" and not os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MERLIN_CPU_SOURCE_READOUT_REQUIRED")
    require(run_id not in {SOURCE_RUN, PREVIOUS_COMPLETION_RUN}, "SEPARATE_SOURCE_READOUT_EXECUTION_REQUIRED")
    runtime = load_runtime(request.get("runtime"), request_path.parent)
    work = Path(request["work"]).resolve()
    require(not work.exists(), "FRESH_SOURCE_READOUT_PUBLICATION_REQUIRED")
    work.mkdir(parents=True)
    report_path = work / "source_readout_publication.json"
    execution = dict(backend="merlin", run_id=run_id, trial_id=trial_id, hostname=socket.gethostname(), allocated_gpus=0)
    state = dict(schema=SOURCE_READOUT_SCHEMA, operation="publish_source_readout", status="VALIDATING_SOURCE_READOUT_INPUTS",
        execution=execution, requested_index=request.get("index"), runtime_sha256=RUNTIME_SHA256,
        source_simulations=0, production_episodes=0, source_execution_reruns=0, readout_execution_reruns=0,
        summary_executions=0, dynamics_executions=0, required_episodes=3000, closure_verified=False,
        consumer_start_authorized=False, published_episodes=0, failures=[], **{field:False for field in QUALIFICATION_FLAGS})

    def report():
        report_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        runtime.publish_json("source_readout_publication_progress.json", state)

    try:
        plan, hashes = qualified_readout_plan(request, runtime)
        source, intake_binding, publisher = single_source_publication_inputs(request, request_path.parent, plan, runtime)
        require(run_id not in {source["source_execution"]["run_id"], publisher["run_id"]},
                "SEPARATE_SOURCE_READOUT_EXECUTION_REQUIRED")
        state.update(source=source, scientific_intake_binding=intake_binding, publication_execution=publisher,
                     publication_kind=request.get("publication_kind", "recovery"),
                     preparation_sha256=request["preparation_sha256"], consumer_sha256=hashes,
                     status="PERSISTING_SOURCE_READOUT")
        report()
        snapshot = persist_source_readout(source, plan, runtime, execution,
                                          native=request.get("publication_kind") == "native_source")
        state.update(status="SOURCE_READOUT_READY", readout_snapshot=snapshot,
                     published_indices=[source["index"]], published_episodes=1)
        report()
        runtime.publish_results([report_path], 0)
        return state
    except Exception as error:
        state.update(status="FAILED_SOURCE_READOUT_PUBLICATION", failures=[dict(error_type=type(error).__name__,
                     message=str(error), traceback=traceback.format_exc())])
        report_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        try:
            runtime.publish_json("source_readout_publication_failure.json", state)
            runtime.publish_results([report_path], 1)
        except Exception as publication_error:
            print("SOURCE_READOUT_FAILURE_PUBLICATION_FAILED:" + str(publication_error), flush=True)
        raise


def prior_1906_readout(binding, base, accepted_binding, accepted, plan, runtime, request):
    path, data = bound_bytes(binding, base)
    wrapper = json.loads(data)
    require(isinstance(wrapper, dict) and isinstance(wrapper.get("uri"), str)
            and wrapper["uri"].startswith("hdfs://")
            and wrapper["uri"].endswith("/" + PREVIOUS_COMPLETION_RUN
                + "/pfc_downstream/observation_completion_progress.json")
            and isinstance(wrapper.get("progress"), dict), "PRIOR_COMPLETION_WRAPPER_CHANGED")
    progress = wrapper["progress"]
    owner = dict(run_id=PREVIOUS_COMPLETION_RUN, trial_id=PREVIOUS_COMPLETION_TRIAL)
    execution = progress.get("execution")
    require(isinstance(execution, dict) and execution.get("backend") == "merlin"
            and execution.get("run_id") == owner["run_id"] and execution.get("trial_id") == owner["trial_id"]
            and execution.get("allocated_gpus") == 0 and progress.get("schema") == SCHEMA
            and progress.get("status") == "WAITING_ORIGINAL2999_SNAPSHOTS"
            and type(progress.get("required_episodes")) is int and progress["required_episodes"] == 3000
            and progress.get("runtime_sha256") == RUNTIME_SHA256
            and progress.get("preparation_sha256") == request["preparation_sha256"]
            and progress.get("consumer_sha256") == request["consumer_sha256"]
            and progress.get("collector_execution") == dict(run_id=COLLECTOR_RUN, trial_id=COLLECTOR_TRIAL)
            and progress.get("accepted_1906_metadata_sha256") == accepted_binding["sha256"]
            and progress.get("failures") == []
            and all(type(progress.get(field)) is int and progress[field] == 0 for field in
                ("source_simulations", "production_episodes", "source_execution_reruns", "readout_execution_reruns"))
            and all(progress.get(field) is False for field in QUALIFICATION_FLAGS),
            "PRIOR_COMPLETION_IDENTITY_OR_SCOPE_CHANGED")
    require(progress.get("source_1906_snapshot") == accepted_source_snapshot(accepted, plan["rows"][1906], runtime),
            "PRIOR1906_ORIGINAL_SOURCE_CHANGED")
    snapshot = validate_single_readout_snapshot(progress.get("recovered_1906_snapshot"), 1906, owner)
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                uri=wrapper["uri"], observed_utc=wrapper.get("observed_utc")), snapshot, owner


def prepare_supplements(request_path):
    """Prepare bound readouts only; keep the running a5 consumer path untouched."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_bytes())
    require(isinstance(request, dict) and request.get("schema") == SCHEMA, "COMPLETION_REQUEST_SCHEMA_INVALID")
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0", "MERLIN_CPU_COMPLETION_REQUIRED")
    require(os.environ["RH2_JOB_RUN_ID"] != PREVIOUS_COMPLETION_RUN, "SEPARATE_PREPARATION_EXECUTION_REQUIRED")
    runtime = load_runtime(request.get("runtime"), request_path.parent)
    work = Path(request["work"]).resolve()
    require(not work.exists(), "FRESH_SUPPLEMENTAL_PREPARATION_REQUIRED")
    work.mkdir(parents=True)
    report_path = work / "supplemental_readout_preparation.json"
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                     hostname=socket.gethostname(), allocated_gpus=0)
    state = dict(schema=SCHEMA, operation="prepare_supplements", status="VALIDATING_SUPPLEMENTAL_READOUT_INPUTS",
                 execution=execution, runtime_sha256=RUNTIME_SHA256, source_simulations=0, production_episodes=0,
                 source_execution_reruns=0, readout_execution_reruns=0, summary_executions=0, dynamics_executions=0,
                 required_episodes=3000, closure_verified=False, supplements=[], failures=[],
                 **{field:False for field in QUALIFICATION_FLAGS})

    def report():
        report_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        runtime.publish_json("supplemental_observation_preparation_progress.json", state)

    try:
        plan, hashes = qualified_readout_plan(request, runtime)
        state["consumer_sha256"] = hashes
        state["preparation_sha256"] = request["preparation_sha256"]
        accepted_path, accepted_data = bound_bytes(request.get("accepted_1906"), request_path.parent)
        accepted = json.loads(accepted_data)
        prior_binding, reused_snapshot, owner = prior_1906_readout(request.get("prior_completion_progress"),
            request_path.parent, request["accepted_1906"], accepted, plan, runtime, request)
        bindings = request.get("accepted_sources")
        require(isinstance(bindings, list) and len(bindings) == len(SUPPLEMENTAL_SOURCE_INDICES),
                "EXACT_SUPPLEMENTAL_SOURCE_SET_REQUIRED")
        sources = [supplemental_source(binding, request_path.parent, plan, runtime) for binding in bindings]
        require(exact_indices([source["index"] for source in sources], SUPPLEMENTAL_SOURCE_INDICES)
                == SUPPLEMENTAL_SOURCE_INDICES, "EXACT_SUPPLEMENTAL_SOURCE_SET_REQUIRED")
        state.update(prior_completion_progress=prior_binding, consumer_owner_execution=owner,
                     accepted_1906_binding=dict(path=str(accepted_path), bytes=len(accepted_data),
                         sha256=hashlib.sha256(accepted_data).hexdigest()),
                     reused_1906_snapshot=reused_snapshot, observation_1906_entry=accepted["observation_entry"])
        require(runtime.restore_snapshot(reused_snapshot) == [1906], "REUSED1906_SNAPSHOT_IDENTITY_CHANGED")
        require(observation_entry(runtime, 1906) == accepted["observation_entry"], "REUSED1906_REPORT_BINDING_CHANGED")
        state["status"] = "PERSISTING_SUPPLEMENTAL_READOUTS"
        report()
        for source in sorted(sources, key=lambda item:item["index"]):
            index = source["index"]
            source["readout_snapshot"] = persist_source_readout(source, plan, runtime, execution)
            state["supplements"].append(source)
            report()
        state.update(status="SUPPLEMENTAL_READOUTS_READY", prepared_indices=sorted(SUPPLEMENTAL_SOURCE_INDICES),
                     available_recovery_indices=sorted(SUPPLEMENTAL_SOURCE_INDICES | {1906}),
                     prepared_episodes=10, reused_persisted_episodes=1,
                     takeover_inputs=dict(runtime=request["runtime"], preparation_sha256=request["preparation_sha256"],
                         consumer_sha256=hashes, collector_progress_uri=request.get("collector_progress_uri"),
                         prior_completion_progress=prior_binding, accepted_1906=state["accepted_1906_binding"],
                         accepted_sources=[dict(source["acceptance_binding"], index=source["index"]) for source in sources],
                         readout_snapshots=[reused_snapshot] + [source["readout_snapshot"] for source in state["supplements"]],
                         consumer_owner_execution=owner, consumer_start_authorized=False, required_episodes=3000))
        report()
        runtime.publish_results([report_path], 0)
        return state
    except Exception as error:
        state.update(status="FAILED_SUPPLEMENTAL_READOUT_PREPARATION", failures=[dict(error_type=type(error).__name__,
                     message=str(error), traceback=traceback.format_exc())])
        report_path.write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        try:
            runtime.publish_json("supplemental_observation_preparation_failure.json", state)
            runtime.publish_results([report_path], 1)
        except Exception as publication_error:
            print("SUPPLEMENTAL_PREPARATION_FAILURE_PUBLICATION_FAILED:" + str(publication_error), flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--prepare-supplements", action="store_true")
    operation.add_argument("--publish-source-readout", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_supplements:
        prepare_supplements(args.request)
    elif args.publish_source_readout:
        publish_source_readout(args.request)
    else:
        run(args.request)


if __name__ == "__main__":
    main()

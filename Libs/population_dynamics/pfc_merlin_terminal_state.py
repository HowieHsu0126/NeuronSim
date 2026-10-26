"""Compare already persisted PFC native and candidate terminal states.

This readout reuses the existing exact and SI field tolerances. It executes no
source simulation. Native refractory sentinels remain in the reported data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile

import numpy as np
from .pfc_deterministic_replay import _array_comparison
from .provenance import freeze_json, sha256_file


def require(value, reason):
    if not value:
        raise ValueError(reason)


def hdfs(*arguments):
    return subprocess.run(["hdfs", "dfs", *map(str, arguments)], check=True, capture_output=True, text=True)


def extract_selected(bundle, files, destination):
    """Read only declared regular files; original bundles remain immutable."""
    destination.mkdir()
    seen = set()
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive:
            if member.name not in files:
                continue
            row = files[member.name]
            require(member.name not in seen and member.isfile() and member.size == row["bytes"],
                    "PFC_STATE_BOUND_MEMBER_INVALID")
            name = Path(row["local_name"])
            require(name.name == str(name) and name.name not in ("", ".", ".."), "PFC_STATE_LOCAL_NAME_INVALID")
            target = destination / name
            stream = archive.extractfile(member)
            require(stream is not None, "PFC_STATE_MEMBER_UNREADABLE")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            require(sha256_file(target) == row["sha256"], "PFC_STATE_BOUND_MEMBER_CHANGED")
            seen.add(member.name)
    require(seen == set(files), "PFC_STATE_BOUND_MEMBERS_MISSING")


def compare_states(actual, expected, case_id):
    """Keep every persistent state field and reuse its existing comparison rule."""
    require(set(actual) == set(expected), "PFC_STATE_FIELDS_DIFFER")
    required = {"E__" + key for key in ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "s_NMDA", "x",
                "I_stim", "lastspike", "not_refractory", "source_indices", "archive_indices", "spike_counts")}
    required |= {"I__" + key for key in ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "lastspike",
                "not_refractory", "source_indices", "archive_indices", "spike_counts")}
    require(set(actual) == required, "PFC_STATE_COMPLETE_SOURCE_FIELDS_REQUIRED")
    results = []
    for name in sorted(actual):
        row = _array_comparison(case_id, "terminal_state", name, actual[name], expected[name])
        row["units"] = ("V" if name.endswith("__v") else "A" if name.endswith("__I_stim")
                        else "s" if name.endswith("__lastspike") else "boolean" if name.endswith("__not_refractory")
                        else "integer" if name.endswith(("__source_indices", "__archive_indices", "__spike_counts"))
                        else "dimensionless")
        results.append(row)
    return results


def checkpoint_path(identity, checkpoint_root, checkpoint_subdirectory=None):
    """Bind the active circuit/block/condition owner before reading archives."""
    require(isinstance(identity, dict), "PFC_STATE_TASK_CHECKPOINT_INVALID")
    circuit, block, condition = (identity.get(name) for name in
                                 ("circuit_index", "block_index", "condition_index"))
    require(type(circuit) is int and 0 <= circuit < 100
            and type(block) is int and block in (6, 7, 8)
            and type(condition) is int and 0 <= condition < 10,
            "PFC_STATE_TASK_CHECKPOINT_INVALID")
    episode = f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{condition:02d}"
    require(identity.get("episode_id") == episode, "PFC_STATE_TASK_CHECKPOINT_INVALID")
    index = 30 * circuit + 10 * (block - 6) + condition
    require(checkpoint_subdirectory is None
            or (type(checkpoint_subdirectory) is str
                and checkpoint_subdirectory == f"terminal_tasks/task_{index:04d}"),
            "PFC_STATE_TASK_CHECKPOINT_INVALID")
    checkpoint = checkpoint_root.rstrip("/")
    if checkpoint_subdirectory is not None:
        checkpoint += "/" + checkpoint_subdirectory
    return checkpoint + "/pfc_terminal_state"


def run(request_path, work, checkpoint_subdirectory=None):
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not work.exists(),
            "PFC_STATE_FRESH_MERLIN_CPU_REQUIRED")
    request = json.loads(request_path.read_bytes())
    require(request.get("schema") == "pfc_persisted_terminal_state_comparison_v1",
            "PFC_STATE_REQUEST_SCHEMA_INVALID")
    checkpoint = checkpoint_path(request.get("identity"), os.environ["RH2_CHECKPOINT_HDFS"],
                                 checkpoint_subdirectory)
    work.mkdir()
    for role in ("native", "candidate"):
        info = request[role]
        archive = work / (role + ".tar.gz")
        hdfs("-get", info["bundle_uri"], archive)
        require(archive.stat().st_size == info["bundle_bytes"] and sha256_file(archive) == info["bundle_sha256"],
                "PFC_STATE_BUNDLE_CHANGED:" + role)
        extract_selected(archive, info["selected_files"], work / role)
        archive.unlink()
    native, candidate = work / "native", work / "candidate"
    first, second = json.loads((native / "report.json").read_bytes()), json.loads((candidate / "report.json").read_bytes())
    case_id = request["identity"]["episode_id"]
    require(first["status"] == "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
            and len(first["cases"]) == 1 and first["cases"][0]["case_id"] == case_id
            and first["resolved_config"] == second["resolved_config"]
            and second["case_id"] == case_id and second["source_execution_complete"] is True
            and second["resolved_config"]["benchmark_episode_candidate"]["identity"] == request["identity"]
            and first["source_entry_sha256"] == second["source_entry_sha256"],
            "PFC_STATE_COMPLETE_NATIVE_CANDIDATE_IDENTITY_CHANGED")
    with np.load(native / "terminal_state.npz", allow_pickle=False) as a, np.load(candidate / "terminal_state.npz", allow_pickle=False) as e:
        rows = compare_states({k: a[k] for k in a.files}, {k: e[k] for k in e.files}, case_id)
    report = dict(schema="pfc_persisted_terminal_state_comparison_v1",
        status="PASS_TERMINAL_STATE_CORRESPONDENCE" if all(r["passed"] for r in rows) else "FAILED_TERMINAL_STATE_CORRESPONDENCE",
        identity=request["identity"], arrays=rows, array_fields=len(rows), passed_fields=sum(r["passed"] for r in rows),
        execution=dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"], hostname=socket.gethostname()),
        source_execution_provenance=request["source_execution_provenance"],
        source_event_comparison_sha256=request["source_event_comparison_sha256"],
        request_sha256=sha256_file(request_path), implementation_sha256=sha256_file(Path(__file__)),
        comparator_sha256=sha256_file(Path(__file__).with_name("pfc_deterministic_replay.py")),
        completed_utc=datetime.now(timezone.utc).isoformat(), source_simulations=0, production_episodes=0,
        full_transition_map_qualified=False, source_qualification_granted=False, scaling_qualified=False,
        scope="same-source completed endpoint states only; not every possible source state or population-law qualification")
    freeze_json(work / "report.json", report)
    hdfs("-mkdir", "-p", checkpoint)
    hdfs("-put", work / "report.json", checkpoint + "/report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint-subdirectory", help="Task prefix within the actual Merlin checkpoint root")
    args = parser.parse_args()
    report = run(args.request, args.work, args.checkpoint_subdirectory)
    print(json.dumps({k: v for k, v in report.items() if k != "arrays"}))
    raise SystemExit(0 if report["status"] == "PASS_TERMINAL_STATE_CORRESPONDENCE" else 1)


if __name__ == "__main__":
    main()

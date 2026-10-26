"""Compose CUGA development execution evidence for the existing native scorer.

Graph termination and process cleanup do not establish task success. This
adapter preserves unscored native state and never supplies a TGC value.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .appworld_benchmark_protocol import EXPERIMENT_SEEDS

CUGA_METHOD = "cuga_full_graph"


def compose_cuga_execution(
    native: Mapping[str, Any], owner: Mapping[str, Any], *,
    native_session_path: str, owner_observation_path: str,
) -> dict[str, Any]:
    """Join native identity with independent owner and graph terminal evidence."""
    if (native.get("schema") != "appworld_native_tool_session_v1"
            or native.get("scope") != "development_tools_only"
            or native.get("formal_eligible") is not False):
        raise ValueError("a development native tool session is required")
    if native.get("split") not in {"train", "dev"}:
        raise ValueError("formal/test CUGA receipts are not admitted here")
    if type(native.get("seed")) is not int or native["seed"] not in EXPERIMENT_SEEDS:
        raise ValueError("CUGA seed must be 41, 42 or 43")
    for field in ("task_id", "run_root", "data_root", "experiment_name", "appworld_version"):
        if not isinstance(native.get(field), str) or not native[field].strip():
            raise ValueError("native session lacks identity field: " + field)
    if not native_session_path or not owner_observation_path:
        raise ValueError("both original evidence paths are required")
    relay = owner.get("relay") or {}
    worker = owner.get("worker") or {}
    cleanup = relay.get("native_cleanup") or {}
    graph = owner.get("graph_observation")
    if graph is not None:
        if not isinstance(graph, Mapping):
            raise ValueError("CUGA graph observation must be an object or absent")
        if graph.get("task_id") != native["task_id"] or graph.get("seed") != native["seed"]:
            raise ValueError("CUGA graph identity differs from the native session")
    worker_not_started = (
        owner.get("status") == "preflight_failed"
        and owner.get("failure_stage") == "preflight"
        and worker.get("spawn_attempted") is False and worker.get("pid") is None
        and isinstance(owner.get("owner_error"), Mapping)
        and graph is None and relay.get("requests") == []
        and owner.get("owner_cancelled") is not True
        and owner.get("cleanup_error") is None and owner.get("relay_error") is None
    )
    terminal = (worker.get("reaped") is True or worker_not_started) and cleanup.get("child_reaped") is True
    native_settled = (
        terminal and type(cleanup.get("exit_code")) is int and cleanup["exit_code"] == 0
        and cleanup.get("termination_signals") == []
        and cleanup.get("request_outcome_unresolved") is False
        and native.get("terminal") is True and native.get("status") == "closed"
        and native.get("native_world_closed") is True and native.get("server_exit_code") == 0
    )
    preflight_rpc_settled = (
        worker_not_started and native_settled
        and owner.get("rpc_outcome_unresolved") is False
        and relay.get("cleanup_error") is None
    )
    rpc_settled = preflight_rpc_settled or (relay.get("status") == "closed"
                   and relay.get("rpc_outcome_unresolved") is False
                   and owner.get("rpc_outcome_unresolved") is False
                   and relay.get("cleanup_error") is None)
    worker_settled = (worker.get("reaped") is True
                      and type(worker.get("exit_code")) is int
                      and worker.get("termination_signals") == [])
    status, reason = "unknown", "process_native_or_rpc_outcome_unresolved"
    graph_status = None if graph is None else graph.get("status")
    if preflight_rpc_settled:
        status, reason = "infrastructure_failure", "owner_preflight_failed_before_worker_start"
    elif native_settled and rpc_settled and worker_settled:
        if graph is None:
            status, reason = "infrastructure_failure", "worker_has_no_graph_observation"
        elif (graph.get("native_cleanup_settled") is not True
              or graph.get("native_cleanup") != cleanup):
            reason = "graph_cleanup_disagrees_with_native_owner"
        elif owner.get("status") not in {"completed", "worker_failed"}:
            reason = "owner_did_not_report_a_settled_worker"
        elif graph_status == "graph_ended" and worker["exit_code"] == 0:
            status, reason = "completed", "source_graph_ended_with_settled_execution"
        elif graph_status in {"interrupted", "awaiting_external_tools", "unfinished"}:
            status, reason = "agent_failure", "source_graph_did_not_finish"
        elif graph_status == "errored" and graph.get("loop"):
            status, reason = "agent_failure", "source_loop_error_with_settled_native_state"
        elif graph_status in {"errored", "unexpected_graph_variant", "graph_ended"}:
            status, reason = "infrastructure_failure", "graph_initialization_or_worker_contract_failure"
        else:
            reason = "graph_terminal_state_unrecognized"
    return {
        "schema": "appworld_development_execution_v1", "scope": "development_only",
        "formal_eligible": False, "method": CUGA_METHOD,
        **{key: native[key] for key in ("split", "task_id", "seed", "run_root", "data_root",
                                        "experiment_name", "appworld_version")},
        "terminal": terminal, "status": status, "execution_status_reason": reason,
        "native_tgc": None, "scoring_status": "not_run", "usage_complete": False,
        "cuga_usage_transport_coverage_qualified": False,
        "cuga_graph_status": graph_status,
        "native_session_path": native_session_path,
        "owner_observation_path": owner_observation_path,
        "native_cleanup": deepcopy(cleanup),
        "rpc_outcome_unresolved": not rpc_settled,
        "cuga_model_usage": deepcopy((graph or {}).get("loop", {}).get("model_usage")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-session", type=Path, required=True)
    parser.add_argument("--owner-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    native_path = args.native_session.resolve(strict=True)
    owner_path = args.owner_observation.resolve(strict=True)
    result = compose_cuga_execution(
        json.loads(native_path.read_text()), json.loads(owner_path.read_text()),
        native_session_path=str(native_path), owner_observation_path=str(owner_path),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

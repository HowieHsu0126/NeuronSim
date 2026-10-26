"""Outcome-blind, animal-grouped Allen cohort selection and immutable splitting."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from .provenance import canonical_json, freeze_json

SESSION_METADATA_URL = (
    "https://api.brain-map.org/api/v2/data/query.json?criteria="
    "model::EcephysSession,rma::include,specimen(donor),well_known_files(well_known_file_type),"
    "rma::options[num_rows$eq'all']"
)


def fetch_metadata(network: dict[str, Any]) -> list[dict[str, Any]]:
    with requests.Session() as session:
        session.trust_env = network["trust_env"]
        response = session.get(SESSION_METADATA_URL, timeout=(
            network["connect_timeout_seconds"], network["read_timeout_seconds"],
        ))
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict) or value.get("success") is not True:
        raise ValueError("ALLEN_METADATA_QUERY_FAILED")
    rows = value.get("msg")
    if not isinstance(rows, list) or value.get("total_rows") != len(rows):
        raise ValueError("ALLEN_METADATA_PAGINATION_INCOMPLETE")
    return rows


def read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if value.get("success") is False:
            raise ValueError("ALLEN_METADATA_QUERY_FAILED")
        rows = value.get("msg")
        if value.get("total_rows") is not None and value["total_rows"] != len(rows or []):
            raise ValueError("ALLEN_METADATA_PAGINATION_INCOMPLETE")
    else:
        rows = value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("SESSION_METADATA_RECORDS_REQUIRED")
    return rows


def _positive_id(value: Any, field: str) -> int:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(f"POSITIVE_SOURCE_ID_REQUIRED:{field}")


def _session_file(row: dict[str, Any], known: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    files = row.get("well_known_files", [])
    if isinstance(files, str):
        files = json.loads(files) if files else []
    candidates = [file for file in files if isinstance(file, dict) and (
        file.get("well_known_file_type", {}).get("name") in {"EcephysNwb", "EcephysNwb2"}
    )]
    if len(candidates) > 1:
        return None, "multiple_session_nwb_files"
    if not candidates and not known:
        if row.get("has_nwb") in (True, "True", "true"):
            return {"file_id": None, "url": None, "expected_size_bytes": None,
                    "resolution": "fetch_single_session_file_metadata_at_intake"}, None
        return None, "session_nwb_missing_or_unverified"
    file = candidates[0] if candidates else known
    file_id = _positive_id(file.get("id", file.get("file_id")), "file_id")
    if known and file_id != known["file_id"]:
        raise ValueError("PREDECLARED_SOURCE_ID_CHANGED")
    size = file.get("size", file.get("file_size", file.get("expected_size_bytes")))
    if known:
        if size is not None and int(size) != known["expected_size_bytes"]:
            raise ValueError("PREDECLARED_SOURCE_SIZE_CHANGED")
        size = known["expected_size_bytes"]
    if size is not None:
        size = _positive_id(size, "expected_size_bytes")
    return {
        "file_id": file_id,
        "url": f"https://api.brain-map.org/api/v2/well_known_file_download/{file_id}",
        "expected_size_bytes": size,
    }, None


def build_cohort(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    policy = config["cohort"]
    normalized = []
    seen = set()
    for row in rows:
        session_id = _positive_id(row.get("id", row.get("ecephys_session_id")), "session_id")
        if session_id in seen:
            raise ValueError(f"DUPLICATE_SESSION_ID:{session_id}")
        seen.add(session_id)
        protocol = row.get("ecephys_session_type", row.get("session_type"))
        reasons = []
        specimen = row.get("specimen") or {}
        raw_specimen = row.get("specimen_id", specimen.get("id"))
        try:
            specimen_id = _positive_id(raw_specimen, "specimen_id")
        except ValueError:
            specimen_id = None
            reasons.append("specimen_identity_missing")
        raw_donor = row.get("donor_id", specimen.get("donor_id", (specimen.get("donor") or {}).get("id")))
        try:
            animal_id = _positive_id(raw_donor, "donor_id")
        except ValueError:
            animal_id = None
            reasons.append("animal_donor_identity_missing")
        if protocol != policy["protocol"]:
            reasons.append("protocol_not_selected")
        known = policy.get("known_sources", {}).get(session_id)
        source, missing = _session_file(row, known)
        if missing:
            reasons.append(missing)
        normalized.append({
            "session_id": session_id, "animal_id": animal_id, "specimen_id": specimen_id, "protocol": protocol,
            "source": source, "included": not reasons, "exclusion_reasons": reasons,
        })
    if len(normalized) != policy["expected_metadata_session_count"]:
        raise ValueError("SOURCE_METADATA_COHORT_INCOMPLETE_OR_CHANGED")
    if sum(row["protocol"] == policy["protocol"] for row in normalized) != policy["expected_protocol_session_count"]:
        raise ValueError("SOURCE_PROTOCOL_COHORT_INCOMPLETE_OR_CHANGED")
    normalized.sort(key=lambda row: row["session_id"])
    anchors = set(policy["exploration_train_session_ids"])
    if not anchors.issubset(seen):
        raise ValueError("PREDECLARED_EXPLORATION_SESSION_MISSING")
    anchor_animals = {row["animal_id"] for row in normalized if row["session_id"] in anchors}
    if None in anchor_animals:
        raise ValueError("EXPLORATION_ANIMAL_IDENTITY_MISSING")
    included_animals = {row["animal_id"] for row in normalized if row["included"]}
    counts = policy["split_counts_by_donor"]
    if len(included_animals) != sum(counts.values()):
        raise ValueError(f"DONOR_QUOTA_COUNT_MISMATCH:available={len(included_animals)}:required={sum(counts.values())}")
    if not anchor_animals.issubset(included_animals) or len(anchor_animals) > counts["train"]:
        raise ValueError("TRAINING_ANCHOR_NOT_ELIGIBLE_FOR_QUOTA")
    ranked = sorted(included_animals - anchor_animals,
        key=lambda animal: hashlib.sha256(f"{policy['split_salt']}:{animal}".encode()).hexdigest())
    split_by_animal = {animal: "train" for animal in anchor_animals}
    offset = 0
    for split in ("train", "validation", "test"):
        count = counts[split] - (len(anchor_animals) if split == "train" else 0)
        split_by_animal.update({animal: split for animal in ranked[offset:offset + count]})
        offset += count
    if offset != len(ranked):
        raise ValueError("DONOR_QUOTA_ALLOCATION_INCOMPLETE")
    for row in normalized:
        row["split"] = split_by_animal.get(row["animal_id"])
    return {
        "schema": "neural_population_cohort_v1",
        "selection": "all_protocol_sessions_before_activity_or_response_analysis",
        "protocol": policy["protocol"],
        "training_seeds": config["training_seeds"],
        "split_policy": {
            "group": "animal_donor_id", "counts": policy["split_counts_by_donor"],
            "method": "sha256_ranked_donor_quotas_with_predeclared_training_anchor",
            "protocol": policy["split_protocol"], "amendment": policy["amendment"],
            "salt": policy["split_salt"], "exploration_train_session_ids": sorted(anchors),
            "anchor_animal_ids": sorted(anchor_animals),
        },
        "metadata_session_count": len(normalized),
        "included_session_count": sum(row["included"] for row in normalized),
        "sessions": normalized,
        "dataset_ready": False,
    }


def freeze_cohort(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cohort = build_cohort(rows, config)
    freeze_json(path, cohort)
    return cohort


def cohort_session(cohort: dict[str, Any], session_id: int) -> dict[str, Any]:
    rows = [row for row in cohort["sessions"] if row["session_id"] == session_id]
    if len(rows) != 1 or not rows[0]["included"]:
        raise ValueError(f"SESSION_NOT_IN_QUALIFIED_METADATA_COHORT:{session_id}")
    # Recheck the group invariant when consuming, not only when constructing.
    animal = rows[0]["animal_id"]
    if any(row["split"] != rows[0]["split"] for row in cohort["sessions"] if row["animal_id"] == animal):
        raise ValueError("ANIMAL_CROSSES_SPLITS")
    return rows[0]


def resolve_source(row: dict[str, Any], network: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    if source["file_id"] is not None:
        return source
    session_id = row["session_id"]
    url = ("https://api.brain-map.org/api/v2/data/query.json?criteria="
           f"model::EcephysSession,rma::criteria,[id$eq{session_id}],"
           "rma::include,specimen(donor),well_known_files(well_known_file_type)")
    with requests.Session() as session:
        session.trust_env = network["trust_env"]
        response = session.get(url, timeout=(network["connect_timeout_seconds"], network["read_timeout_seconds"]))
        response.raise_for_status()
        value = response.json()
    records = value.get("msg") if isinstance(value, dict) and value.get("success") is True else None
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("SINGLE_SESSION_SOURCE_METADATA_REQUIRED")
    record = records[0]
    if (_positive_id(record.get("id"), "session_id") != session_id
            or _positive_id(record.get("specimen_id"), "specimen_id") != row["specimen_id"]
            or _positive_id((record.get("specimen") or {}).get("donor_id"), "donor_id") != row["animal_id"]
            or record.get("ecephys_session_type", record.get("session_type")) != row["protocol"]):
        raise ValueError("RESOLVED_SOURCE_IDENTITY_MISMATCH")
    resolved, reason = _session_file(record, None)
    if reason or resolved["file_id"] is None:
        raise ValueError("SESSION_NWB_FILE_ID_NOT_RESOLVED")
    return resolved

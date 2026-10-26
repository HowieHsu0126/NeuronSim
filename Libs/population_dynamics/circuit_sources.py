"""Fetch author source at resolved commits before native circuit reproduction.

This command acquires code and records unresolved model assets. Downloading
source never qualifies its numerical behavior or the scientific dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

import requests

from .provenance import freeze_json, sha256_file

SOURCES = {
    "v1_model_index": ("ModelDBRepository/265592", "main"),
    "v1_network_builder": ("AllenInstitute/biorealistic-v1-model", "master"),
    "pfc_working_memory": ("EPFL-LCN/neuronaldynamics-exercises", "master"),
    "ca3_formation_retrieval": ("jkopsick/simulate_formation_and_retrieval", "main"),
    "carlsim_ca3": ("UCI-CARL/CARLsim6", "feat/ca3net"),
    "cuda_samples": ("NVIDIA/cuda-samples", "v11.8"),
}
MAX_CODE_BYTES = 8 * 1024 * 1024


def materialize_build_inputs(source: Path, paths: list[str]) -> dict:
    """Fetch named build inputs against the already-pinned Git tree identities."""
    manifest = json.loads((source / "source.json").read_text())
    expected = {row["path"]: row for row in manifest["deferred_blobs"]}
    session = requests.Session()
    session.trust_env = False
    repository = manifest["repository"].removeprefix("https://github.com/")
    completed = []
    for name in paths:
        row = expected[name]
        target = source / name
        if target.exists():
            payload = target.read_bytes()
        else:
            response = session.get(f"https://raw.githubusercontent.com/{repository}/{manifest['revision']}/{name}", timeout=(15,120))
            response.raise_for_status()
            payload = response.content
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob != row["sha"] or len(payload) != row["size"]:
            raise ValueError("BUILD_INPUT_BLOB_MISMATCH:" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        completed.append(dict(path=name, bytes=len(payload), git_blob_sha1=blob))
    session.close()
    return dict(source_revision=manifest["revision"], materialized_inputs=completed)


def fetch_source(name: str, destination: Path) -> dict:
    repository, requested_ref = SOURCES[name]
    destination.mkdir(parents=True, exist_ok=True)
    existing = destination / "source.json"
    if existing.exists():
        report = json.loads(existing.read_text())
        for row in report["files"]:
            if sha256_file(destination / row["path"]) != row["sha256"]:
                raise ValueError("PINNED_SOURCE_CHANGED:" + row["path"])
        return report
    session = requests.Session()
    session.trust_env = False
    session.headers["User-Agent"] = "NeuronSim-source-reproduction"
    api = "https://api.github.com/repos/" + repository

    def get(url):
        response = session.get(url, timeout=(15, 120))
        response.raise_for_status()
        return response

    started = time.time()
    revision = get(api + "/commits/" + requested_ref).json()["sha"]
    tree = get(api + "/git/trees/" + revision + "?recursive=1").json()
    if tree.get("truncated"):
        raise ValueError("SOURCE_TREE_TRUNCATED")
    files, deferred, links = [], [], []
    for row in tree["tree"]:
        if row["type"] != "blob":
            continue
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("SOURCE_PATH_INVALID")
        if name == "cuda_samples" and not (str(relative).startswith("Common/") or relative.name == "LICENSE"):
            continue
        # Figures, notebooks and precomputed simulation products are not code.
        code = (relative.suffix.lower() in {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu",
                    ".cuh", ".json", ".yaml", ".yml", ".csv", ".txt", ".md", ".rst",
                    ".html", ".sh", ".mk", ".cmake"}
                or relative.name.lower().startswith(("license", "copying", "makefile", "cmakelists")))
        if not code or row.get("size", MAX_CODE_BYTES + 1) > MAX_CODE_BYTES:
            deferred.append({k: row[k] for k in ("path", "sha", "size") if k in row})
            continue
        payload = get(f"https://raw.githubusercontent.com/{repository}/{revision}/{relative}").content
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob != row["sha"] or len(payload) != row["size"]:
            raise ValueError("SOURCE_BLOB_MISMATCH:" + str(relative))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError("SOURCE_FILE_CONFLICT:" + str(relative))
        if not target.exists():
            target.write_bytes(payload)
        files.append(dict(path=str(relative), bytes=len(payload), git_blob_sha1=blob,
                          sha256=hashlib.sha256(payload).hexdigest()))
        if relative.suffix.lower() in {".md", ".html", ".rst", ".txt"}:
            for link in re.findall(r"https?://[^\s<>\"']+", payload.decode("utf-8", errors="replace")):
                if any(part in link.lower() for part in ("dropbox", "zenodo", "carlsim", "allen", "bmtk")):
                    links.append(dict(source_file=str(relative), url=link))
    report = dict(source_id=name, repository="https://github.com/" + repository,
                  requested_ref=requested_ref, revision=revision, files=files,
                  deferred_blobs=deferred, discovered_asset_links=links,
                  license_files=[r["path"] for r in files if PurePosixPath(r["path"]).name.lower().startswith(("license", "copying"))],
                  source_code_acquired=True, native_reproduction_completed=False,
                  scientific_reference_qualified=False, elapsed_seconds=time.time() - started)
    freeze_json(existing, report)
    session.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "all"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-input", action="append", default=[])
    args = parser.parse_args()
    names = list(SOURCES) if args.source == "all" else [args.source]
    if args.build_input:
        if args.source == "all":
            parser.error("named build inputs require one source")
        print(json.dumps(materialize_build_inputs(args.output / args.source, args.build_input)))
        return
    failures = []
    for name in names:
        try:
            report = fetch_source(name, args.output / name)
            print(json.dumps({k: report[k] for k in ("source_id", "revision", "license_files", "elapsed_seconds")}), flush=True)
        except (requests.RequestException, ValueError, KeyError, OSError) as error:
            failures.append(dict(source_id=name, error_type=type(error).__name__, error=str(error)))
            print(json.dumps(failures[-1]), flush=True)
    if failures:
        args.output.mkdir(parents=True, exist_ok=True)
        freeze_json(args.output / "intake_failures.json", failures)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

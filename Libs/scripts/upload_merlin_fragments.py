#!/usr/bin/env python3
"""Upload one immutable archive through at most three independent Merlin uploads.

Run on a compute host with --source and --scratch on local filesystems. Metadata
may reside in the existing operation directory. Each fragment keeps its own
resumable upload manifest; --manifest is published only for a complete upload.
This command does not produce simulation episodes or alter archive contents.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile
from typing import Any, Iterator


MIB = 1024 * 1024
LOCAL_FILESYSTEMS = {
    "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "bcachefs",
    "tmpfs", "ramfs", "overlay", "aufs", "rootfs", "reiserfs", "jfs",
}


class TransportError(RuntimeError):
    """A failure whose message contains only local transport diagnostics."""


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise TransportError(f"Refusing a symlink manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise TransportError(f"Cannot read a valid JSON manifest: {path}") from None
    if not isinstance(value, dict):
        raise TransportError(f"Manifest must contain an object: {path}")
    return value


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Publish without replacing an existing, different manifest."""
    if path.exists() or path.is_symlink():
        if read_json(path) != value:
            raise TransportError(f"Existing manifest conflicts with this upload: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != value:
                raise TransportError(f"Existing manifest conflicts with this upload: {path}")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_local(path: Path) -> None:
    """Resolve the containing Linux mount; unknown filesystems are not scratch."""
    resolved = path.resolve()
    best: tuple[int, str] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
        for line in lines:
            left, right = line.split(" - ", 1)
            mount = re.sub(
                r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)),
                left.split()[4],
            )
            mount_path = Path(mount)
            if resolved == mount_path or mount_path in resolved.parents:
                candidate = (len(mount_path.parts), right.split()[0])
                if best is None or candidate[0] > best[0]:
                    best = candidate
    except (OSError, ValueError, IndexError):
        raise TransportError("Cannot establish the local scratch filesystem.") from None
    if best is None or best[1] not in LOCAL_FILESYSTEMS:
        filesystem = best[1] if best is not None else "unknown"
        raise TransportError(f"A verified local filesystem is required: {path} ({filesystem})")


def snapshot(info: os.stat_result) -> tuple[int, ...]:
    if not stat.S_ISREG(info.st_mode):
        raise TransportError("The input archive and fragments must be regular files.")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def scan_source(
    source: Path, chunk_bytes: int, workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, ...]]:
    """Record global, fragment, and chunk identities in one sequential read."""
    initial = snapshot(source.stat())
    size = initial[2]
    if size <= 0:
        raise TransportError("The input archive is empty.")
    chunk_count = (size + chunk_bytes - 1) // chunk_bytes
    fragment_count = min(workers, chunk_count)
    quotient, remainder = divmod(chunk_count, fragment_count)
    whole = hashlib.sha256()
    fragments: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    offset = 0
    with source.open("rb") as stream:
        if snapshot(os.fstat(stream.fileno())) != initial:
            raise TransportError("The input archive changed before scanning.")
        for index in range(fragment_count):
            count = quotient + int(index < remainder)
            fragment_bytes = min(count * chunk_bytes, size - offset)
            digest = hashlib.sha256()
            remaining = fragment_bytes
            while remaining:
                length = min(chunk_bytes, remaining)
                data = stream.read(length)
                if len(data) != length:
                    raise TransportError("The input archive changed while scanning.")
                digest.update(data)
                whole.update(data)
                chunks.append({"bytes": length, "sha256": hashlib.sha256(data).hexdigest()})
                remaining -= length
            fragments.append({
                "index": index, "offset": offset, "bytes": fragment_bytes,
                "sha256": digest.hexdigest(),
            })
            offset += fragment_bytes
        if stream.read(1) or snapshot(os.fstat(stream.fileno())) != initial:
            raise TransportError("The input archive changed while scanning.")
    if snapshot(source.stat()) != initial:
        raise TransportError("The input archive changed while scanning.")
    return ({
        "file": str(source), "bytes": size, "sha256": whole.hexdigest(),
        "chunk_bytes": chunk_bytes, "fragments": fragments,
    }, chunks, initial)


def verify_fragment(path: Path, expected: dict[str, Any], chunk_bytes: int) -> None:
    if path.is_symlink():
        raise TransportError(f"Refusing a symlink fragment: {path}")
    initial = snapshot(path.stat())
    if initial[2] != expected["bytes"]:
        raise TransportError(f"Fragment size conflicts with its recorded identity: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if snapshot(os.fstat(stream.fileno())) != initial:
            raise TransportError(f"Fragment changed before verification: {path}")
        while data := stream.read(chunk_bytes):
            digest.update(data)
        if snapshot(os.fstat(stream.fileno())) != initial:
            raise TransportError(f"Fragment changed while verifying: {path}")
    if snapshot(path.stat()) != initial or digest.hexdigest() != expected["sha256"]:
        raise TransportError(f"Fragment conflicts with its recorded identity: {path}")


def prepare_fragment(
    source: Path, path: Path, expected: dict[str, Any], chunk_bytes: int,
) -> None:
    if path.exists() or path.is_symlink():
        verify_fragment(path, expected, chunk_bytes)
        return
    temporary: Path | None = None
    try:
        with source.open("rb") as incoming, tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as outgoing:
            temporary = Path(outgoing.name)
            incoming.seek(expected["offset"])
            remaining = expected["bytes"]
            digest = hashlib.sha256()
            while remaining:
                length = min(chunk_bytes, remaining)
                data = incoming.read(length)
                if len(data) != length:
                    raise TransportError("The input archive changed while splitting.")
                outgoing.write(data)
                digest.update(data)
                remaining -= length
            if digest.hexdigest() != expected["sha256"]:
                raise TransportError("The input archive changed while splitting.")
            outgoing.flush()
            os.fsync(outgoing.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify_fragment(path, expected, chunk_bytes)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_upload(
    value: dict[str, Any], source: Path, expected: dict[str, Any],
    chunk_bytes: int, expected_chunks: list[dict[str, Any]], *, complete: bool,
) -> None:
    if (
        value.get("file") != str(source)
        or type(value.get("bytes")) is not int
        or value["bytes"] != expected["bytes"]
        or type(value.get("chunk_bytes")) is not int
        or value["chunk_bytes"] != chunk_bytes
        or type(value.get("complete")) is not bool
    ):
        raise TransportError(f"Upload manifest identity conflicts with its input: {source}")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or len(chunks) > len(expected_chunks):
        raise TransportError(f"Invalid upload chunk list: {source}")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or (
            type(chunk.get("index")) is not int or chunk["index"] != index
            or type(chunk.get("bytes")) is not int
            or chunk["bytes"] != expected_chunks[index]["bytes"]
            or chunk.get("sha256") != expected_chunks[index]["sha256"]
            or not isinstance(chunk.get("hdfs_path"), str) or not chunk["hdfs_path"]
        ):
            raise TransportError(f"Upload chunk {index} conflicts with its input: {source}")
    if "sha256" in value and value["sha256"] != expected["sha256"]:
        raise TransportError(f"Upload hash conflicts with its input: {source}")
    if complete or value["complete"]:
        if (
            value["complete"] is not True or len(chunks) != len(expected_chunks)
            or value.get("sha256") != expected["sha256"]
        ):
            raise TransportError(f"Upload is not complete and verified: {source}")


def load_helpers(path: Path) -> Any:
    name = "neuronsim_merlin_fragment_helpers"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise TransportError("Cannot load the existing Merlin upload helpers.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def upload_fragment(
    helpers: Any, api_path: Path, fragment: Path, manifest: Path, work: Path,
    expected: dict[str, Any], expected_chunks: list[dict[str, Any]],
    chunk_mib: int, timeout: int,
) -> dict[str, Any]:
    chunk_bytes = chunk_mib * MIB
    try:
        verify_fragment(fragment, expected, chunk_bytes)
        if manifest.exists() or manifest.is_symlink():
            saved = read_json(manifest)
            validate_upload(saved, fragment, expected, chunk_bytes, expected_chunks, complete=False)
            if saved["complete"]:
                return saved
        # Each worker owns its module and adapter closure. Credential headers stay
        # on stdin inside the existing curl adapter; no API responses are logged.
        api = helpers.module_at(f"merlin_fragment_{expected['index']}", api_path)

        def upload_chunk(data: bytes, name: str, timeout: int = timeout) -> Any:
            return helpers.curl_upload_chunk(api, work, data, name, timeout=timeout)

        api.upload_chunk = upload_chunk
        result = api.upload(fragment, manifest, chunk_mib)
        if not isinstance(result, dict):
            raise TransportError(f"Fragment {expected['index']} returned an invalid upload result.")
        validate_upload(result, fragment, expected, chunk_bytes, expected_chunks, complete=True)
        verify_fragment(fragment, expected, chunk_bytes)
        saved = read_json(manifest)
        validate_upload(saved, fragment, expected, chunk_bytes, expected_chunks, complete=True)
        if saved != result:
            raise TransportError(f"Fragment {expected['index']} result differs from its saved manifest.")
        return saved
    except TransportError:
        raise
    except Exception as error:
        # Only the local curl adapter's bounded stderr is safe to report.
        # Other upstream errors can contain headers or response bodies.
        detail = type(error).__name__
        if isinstance(error, RuntimeError) and str(error).startswith("Curl upload failed: curl:"):
            detail = str(error).splitlines()[0][:240]
        raise TransportError(
            f"Fragment {expected['index']} failed ({detail}); "
            f"resume state remains at {manifest}"
        ) from None


@contextmanager
def operation_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TransportError("Another process owns this fragment upload operation.") from None
        yield
    finally:
        os.close(descriptor)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if any(name.split(".")[0].lower() == "mgmt2" for name in (socket.gethostname(), os.uname().nodename)):
        raise TransportError("Archive hashing, splitting, and upload are forbidden on mgmt2.")
    source = arguments.source.resolve(strict=True)
    scratch = arguments.scratch.resolve()
    manifest_argument = arguments.manifest.absolute()
    if manifest_argument.is_symlink():
        raise TransportError("The merged manifest must not be a symlink.")
    manifest = manifest_argument.resolve()
    if manifest == source:
        raise TransportError("The merged manifest must not replace the input archive.")
    require_local(source)
    require_local(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata = manifest.parent / f"{manifest.name}.fragments"
    if metadata.is_symlink():
        raise TransportError("The fragment metadata directory must not be a symlink.")
    metadata.mkdir(exist_ok=True)
    operation_id = hashlib.sha256(str(manifest).encode("utf-8")).hexdigest()[:16]
    data_directory = scratch / f"merlin-fragments-{operation_id}"
    if data_directory.is_symlink():
        raise TransportError("The fragment scratch directory must not be a symlink.")
    data_directory.mkdir(exist_ok=True)
    require_local(data_directory)
    chunk_bytes = arguments.chunk_mib * MIB

    with operation_lock(metadata / "operation.lock"):
        identity, source_chunks, initial = scan_source(source, chunk_bytes, arguments.workers)
        if manifest.exists():
            existing = read_json(manifest)
            validate_upload(existing, source, identity, chunk_bytes, source_chunks, complete=True)
            return {"manifest": str(manifest), "bytes": identity["bytes"],
                    "chunks": len(source_chunks), "complete": True, "reused": True}
        identity["scratch"] = str(data_directory)
        publish_json(metadata / "source_identity.json", identity)
        prepared: list[tuple[dict[str, Any], Path, Path, Path, list[dict[str, Any]]]] = []
        for fragment in identity["fragments"]:
            index = fragment["index"]
            path = data_directory / f"fragment-{index:03d}.bin"
            state = metadata / f"fragment-{index:03d}.upload.json"
            work = data_directory / f"curl-{index:03d}"
            if work.is_symlink():
                raise TransportError("A curl scratch directory must not be a symlink.")
            work.mkdir(exist_ok=True)
            require_local(work)
            prepare_fragment(source, path, fragment, chunk_bytes)
            start = fragment["offset"] // chunk_bytes
            count = (fragment["bytes"] + chunk_bytes - 1) // chunk_bytes
            chunks = source_chunks[start:start + count]
            if state.exists() or state.is_symlink():
                validate_upload(read_json(state), path, fragment, chunk_bytes, chunks, complete=False)
            prepared.append((fragment, path, state, work, chunks))
        if snapshot(source.stat()) != initial:
            raise TransportError("The input archive changed during preparation.")

        helpers = load_helpers(arguments.stage_helper.resolve(strict=True))
        api_path = arguments.merlin_io.resolve(strict=True)
        results: dict[int, dict[str, Any]] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = {
                pool.submit(upload_fragment, helpers, api_path, path, state, work,
                            fragment, chunks, arguments.chunk_mib, arguments.timeout): fragment["index"]
                for fragment, path, state, work, chunks in prepared
            }
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except TransportError as error:
                    failures.append(str(error))
        if failures:
            raise TransportError("; ".join(failures))

        checked, _, final_snapshot = scan_source(source, chunk_bytes, arguments.workers)
        checked["scratch"] = str(data_directory)
        if final_snapshot != initial or checked != identity:
            raise TransportError("The input archive changed; the merged manifest was withheld.")
        merged_chunks: list[dict[str, Any]] = []
        for fragment, path, _, _, _ in prepared:
            verify_fragment(path, fragment, chunk_bytes)
            for chunk in results[fragment["index"]]["chunks"]:
                merged_chunks.append({
                    "index": len(merged_chunks), "bytes": chunk["bytes"],
                    "sha256": chunk["sha256"], "hdfs_path": chunk["hdfs_path"],
                })
        merged = {
            "file": str(source), "bytes": identity["bytes"], "chunk_bytes": chunk_bytes,
            "chunks": merged_chunks, "complete": True, "sha256": identity["sha256"],
        }
        validate_upload(merged, source, identity, chunk_bytes, source_chunks, complete=True)
        publish_json(manifest, merged)
        return {"manifest": str(manifest), "bytes": identity["bytes"],
                "chunks": len(merged_chunks), "fragments": len(prepared),
                "complete": True, "reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Immutable local transport archive")
    parser.add_argument("--scratch", type=Path, required=True, help="Local compute-host fragment scratch")
    parser.add_argument("--manifest", type=Path, required=True, help="Complete merged upload manifest")
    parser.add_argument("--merlin-io", type=Path, required=True, help="Installed merlin_io.py")
    parser.add_argument("--stage-helper", type=Path,
                        default=Path(__file__).with_name("stage_v1_merlin_block_inputs.py"))
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--timeout", type=int, default=180, help="curl adapter timeout in seconds")
    arguments = parser.parse_args()
    if arguments.chunk_mib < 1 or arguments.timeout < 1:
        parser.error("--chunk-mib and --timeout must be positive integers")
    try:
        result = run(arguments)
    except TransportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; fragment manifests remain available for resumption.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: transport preparation failed ({type(error).__name__}); "
              "existing fragment states were preserved.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded HTTPS NWB transfer; incomplete bytes never acquire a complete identity."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import requests

from .provenance import freeze_json, sha256_file

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def verify_local_nwb(path: Path, expected_size: int | None) -> dict[str, Any]:
    if expected_size is None:
        raise ValueError("SOURCE_SIZE_UNVERIFIED")
    if path.suffix == ".part" or not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"NWB_SIZE_MISMATCH:{path}")
    with path.open("rb") as stream:
        if stream.read(8) != HDF5_SIGNATURE:
            raise ValueError("NWB_HDF5_SIGNATURE_MISMATCH")
    return {"size_bytes": expected_size, "sha256": sha256_file(path)}


def download_nwb(source: dict[str, Any], cache_dir: Path, network: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if not source["url"].startswith("https://"):
        raise ValueError("HTTPS_SOURCE_REQUIRED")
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / f"allen_ecephys_{source['file_id']}.nwb"
    partial = final.with_suffix(".nwb.part")
    identity = final.with_suffix(".download.json")
    # This record is consumed on resume to prevent appending a different source.
    freeze_json(identity, {"file_id": source["file_id"], "url": source["url"]})
    expected = source.get("expected_size_bytes")
    complete_record = final.with_suffix(".complete.json")
    if final.exists():
        previous = json.loads(complete_record.read_text()) if complete_record.exists() else None
        if expected is None and previous is not None:
            expected = previous["size_bytes"]
        checked = verify_local_nwb(final, expected)
        if previous is not None and checked != previous:
            raise ValueError("COMPLETE_NWB_CONTENT_CHANGED")
        freeze_json(complete_record, checked)
        return final, checked
    offset = partial.stat().st_size if partial.exists() else 0
    if expected is not None and offset > expected:
        raise ValueError("PARTIAL_NWB_EXCEEDS_SOURCE_SIZE")
    if expected is not None and offset == expected:
        # A previous transfer may have completed before finalization. Validate the
        # HDF5 container before assigning the complete filename below.
        pass
    else:
        with requests.Session() as session:
            session.trust_env = network["trust_env"]
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with session.get(source["url"], headers=headers, stream=True, timeout=(
                network["connect_timeout_seconds"], network["read_timeout_seconds"],
            )) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset:
                        raise ValueError("DOWNLOAD_RANGE_MISMATCH")
                    total = int(match[3])
                elif response.status_code == 200:
                    if offset:
                        raise ValueError("DOWNLOAD_RESUME_NOT_HONORED")
                    total = int(response.headers.get("Content-Length", "0"))
                else:
                    raise ValueError(f"DOWNLOAD_STATUS_UNSUPPORTED:{response.status_code}")
                if total <= 0 or (expected is not None and expected != total):
                    raise ValueError("DOWNLOAD_SOURCE_SIZE_CHANGED_OR_UNKNOWN")
                expected = total
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            if stream.tell() + len(block) > expected:
                                raise ValueError("DOWNLOAD_EXCEEDS_DECLARED_SIZE")
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
    if expected is None or partial.stat().st_size != expected:
        raise ValueError("DOWNLOAD_INCOMPLETE")
    # HDF5 opening verifies the superblock's declared EOF, not just its prefix.
    import h5py
    with h5py.File(partial, "r") as handle:
        if "nwb_version" not in handle.attrs:
            raise ValueError("NWB_VERSION_MISSING")
    partial.rename(final)
    checked = verify_local_nwb(final, expected)
    freeze_json(complete_record, checked)
    return final, checked


def _metadata_deadline(network) -> float:
    total = network.get("metadata_total_timeout_seconds", 120)
    if type(total) not in (int, float) or not math.isfinite(total) or total <= 0:
        raise ValueError("METADATA_TOTAL_TIMEOUT_MUST_BE_FINITE_POSITIVE")
    return total


def _remote_identity(filesystem, url: str, expected_size: int, expected_etag: str, network):
    # FSSpec runs the HTTP metadata coroutine on its maintained event loop. The
    # sync timeout bounds the complete HEAD/GET metadata operation, including
    # DNS and fallback, rather than only individual blocking socket reads.
    total = _metadata_deadline(network)
    info = filesystem.info(url, timeout=total)
    size = info.get("size")
    etag = str(info.get("ETag", "")).strip('"')
    if size != expected_size:
        raise ValueError(f"REMOTE_SOURCE_SIZE_MISMATCH:{size}:{expected_size}")
    if not etag or etag != expected_etag.strip('"'):
        raise ValueError("REMOTE_SOURCE_ETAG_MISMATCH")
    return {"access_url":url,"size_bytes":size,"etag":etag,
            "source_full_asset_downloaded":False}


@contextmanager
def open_remote_nwb(url: str, expected_size: int, expected_etag: str, network, *, progress=None):
    """Version-bound HTTP range access to the configured Allen NWB object."""
    import aiohttp
    import fsspec
    import h5py

    if not url.startswith("https://") or not expected_etag or expected_size <= 0:
        raise ValueError("VERIFIED_REMOTE_SOURCE_IDENTITY_REQUIRED")
    total = _metadata_deadline(network)
    metadata_fs = fsspec.filesystem("http", client_kwargs={
        "trust_env":network["trust_env"],
        "timeout":aiohttp.ClientTimeout(total=total,connect=network["connect_timeout_seconds"]),
    })
    identity = _remote_identity(metadata_fs,url,expected_size,expected_etag,network)
    if progress:
        progress({"event":"REMOTE_IDENTITY_VERIFIED", **identity})
    def etag_reader():
        return _remote_identity(metadata_fs,url,expected_size,expected_etag,network)["etag"]
    range_total = network.get("range_total_timeout_seconds", 120)
    block_size = network.get("range_block_size_bytes", 64 * 1024)
    cache_blocks = network.get("range_cache_blocks", 128)
    if type(range_total) not in (int, float) or not math.isfinite(range_total) or range_total <= 0:
        raise ValueError("RANGE_TOTAL_TIMEOUT_MUST_BE_FINITE_POSITIVE")
    if any(type(value) is not int or value <= 0 for value in (block_size, cache_blocks)):
        raise ValueError("RANGE_CACHE_LIMITS_MUST_BE_POSITIVE_INTEGERS")
    # Sparse HDF5 metadata needs small on-demand ranges. Speculative 4-MiB
    # background reads caused multi-minute opens on the measured source route.
    # A per-socket idle timeout alone did not bound those slowly arriving bodies.
    timeout = aiohttp.ClientTimeout(total=range_total,connect=network["connect_timeout_seconds"],
                                   sock_read=network["read_timeout_seconds"])
    fs = fsspec.filesystem("http",client_kwargs={"trust_env":network["trust_env"],"timeout":timeout})
    with fs.open(url,"rb",size=expected_size,block_size=block_size,
                 cache_type="blockcache",cache_options={"maxblocks":cache_blocks}) as stream:
        with h5py.File(stream,"r",rdcc_nbytes=16*1024*1024) as handle:
            yield handle,identity,etag_reader
            etag_reader()

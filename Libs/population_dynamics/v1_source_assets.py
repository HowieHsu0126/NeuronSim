"""Resolve V1 assets from Allen's public model page without guessing filenames."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.parse import unquote, urlparse

import requests

from .provenance import freeze_json

MODEL_PAGE = "https://brain-map.org/our-research/computational-modeling/models-of-the-mouse-primary-visual-cortex"


def folder_urls(page: str) -> list[str]:
    """Extract links already delivered in the public folder page's prefetch."""
    result = set()
    page = page.replace('\\"', '"')
    for match in re.finditer(r'registerStreamedPrefetch\("([^"]+)"\s*,\s*"([^"]+)"', page):
        key = base64.b64decode(match[1] + "===")
        if b"FolderEntries" not in key:
            continue
        value = base64.b64decode(match[2] + "===")
        for url in re.findall(rb'https://www\.dropbox\.com/scl/fo/[A-Za-z0-9_/%.-]+\?rlkey=[A-Za-z0-9]+&dl=0', value):
            result.add(url.decode("ascii"))
    return sorted(result)


def published_child(parent_index: Path, name: str) -> str:
    urls = folder_urls(parent_index.read_text())
    # Published listings include the current folder as well as its children.
    # Use the existing intake receipt to exclude that self-link when names repeat.
    receipt_path = parent_index.with_name("source.json")
    parent_url = None
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("artifact") == parent_index.name:
            parent_url = receipt["published_url"]
    matches = [url for url in urls if url != parent_url
               and unquote(urlparse(url).path.split("/")[-1]) == name]
    if len(matches) != 1:
        raise ValueError("ONE_PUBLISHED_CHILD_ASSET_REQUIRED:" + name)
    return matches[0]


def fetch_child(parent_index: Path, name: str, output: Path, *, text_asset=False):
    if (output / "source.json").exists():
        raise FileExistsError("V1_SOURCE_RECEIPT_ALREADY_EXISTS")
    published = published_child(parent_index, name)
    url = published.replace("&dl=0", "&dl=1") if text_asset else published
    output.mkdir(parents=True, exist_ok=True)
    limit = 1024*1024 if text_asset else 8*1024*1024
    with requests.get(url, stream=True, timeout=(15,45)) as response:
        response.raise_for_status()
        pieces, size = [], 0
        for part in response.iter_content(65536):
            size += len(part)
            if size > limit:
                raise ValueError("SOURCE_METADATA_READ_LIMIT_EXCEEDED")
            pieces.append(part)
        payload = b"".join(pieces)
        text = payload.decode("utf-8")
        target = output / ("source_asset.txt" if text_asset else "asset_index.html")
        target.write_bytes(payload)
        result = dict(parent_index_sha256=hashlib.sha256(parent_index.read_bytes()).hexdigest(),
                      published_url=published, requested_url=url, resolved_url=response.url,
                      artifact=target.name, sha256=hashlib.sha256(payload).hexdigest(), bytes=size,
                      observed_child_urls=[] if text_asset else folder_urls(text),
                      model_assets_acquired=False, complete_listing_qualified=False)
    freeze_json(output / "source.json", result)
    return result


def resolve_download(parent_index: Path, name: str, output: Path):
    """Resolve headers only, so a control plane never relays the asset body."""
    published = published_child(parent_index, name)
    output.mkdir(parents=True, exist_ok=False)
    with requests.get(published.replace("&dl=0", "&dl=1"), stream=True, timeout=(15, 45)) as response:
        response.raise_for_status()
        if "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("V1_DOWNLOAD_RESOLVED_TO_HTML")
        size = response.headers.get("Content-Length")
        if size is None or not size.isdecimal() or int(size) <= 0:
            raise ValueError("V1_DOWNLOAD_SIZE_UNRESOLVED")
        result = dict(published_url=published, resolved_url=response.url, artifact=name,
                      expected_bytes=int(size), etag=response.headers.get("ETag"),
                      parent_index_sha256=hashlib.sha256(parent_index.read_bytes()).hexdigest(),
                      asset_body_acquired=False)
    freeze_json(output / "download.json", result)
    return result


METADATA_BYTE_LIMIT = 16 * 1024 * 1024
METADATA_ATTEMPT_SECONDS = 240
DEFAULT_RANGE_BYTES = 512 * 1024
LOCAL_FILESYSTEMS = {"xfs", "ext2", "ext3", "ext4", "btrfs", "tmpfs", "zfs", "overlay", "f2fs"}


def _expected_checksums(source: dict) -> dict[str, str]:
    checksums = {}
    for algorithm, length in (("sha256", 64), ("md5", 32)):
        key = "expected_" + algorithm
        if key in source:
            value = source[key]
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{%d}" % length, value) is None:
                raise ValueError("V1_INVALID_EXPECTED_" + algorithm.upper())
            checksums[algorithm] = value.lower()
    return checksums


def _strong_etag(source: dict) -> bool:
    value = source.get("etag")
    return isinstance(value, str) and bool(value) and not value.startswith("W/")


class _ResolvedAddressAdapter(requests.adapters.HTTPAdapter):
    """Route one HTTPS origin to an explicit IP while authenticating its name.

    The request URL is unchanged. Both TLS SNI and certificate matching use
    the source hostname, and HTTP Host retains the original authority. The
    adapter belongs only to the download session; redirects outside that
    origin and proxy transport are rejected before opening a connection.
    """

    def __init__(self, url: str, address: str):
        parsed = urlparse(url)
        hostname = parsed.hostname
        try:
            port = parsed.port or 443
        except ValueError:
            raise ValueError("V1_RESOLVE_REQUIRES_HTTPS_SOURCE_HOST") from None
        if (parsed.scheme != "https" or not hostname or parsed.username is not None
                or parsed.password is not None or len(hostname) > 253
                or any(re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label) is None
                       for label in hostname.split("."))):
            raise ValueError("V1_RESOLVE_REQUIRES_HTTPS_SOURCE_HOST")
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise ValueError("V1_RESOLVE_REQUIRES_HTTPS_SOURCE_HOST")
        if not isinstance(address, str):
            raise ValueError("V1_RESOLVE_REQUIRES_IP_ADDRESS")
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise ValueError("V1_RESOLVE_REQUIRES_IP_ADDRESS") from None
        if ip.is_unspecified or ip.is_multicast or "%" in str(ip):
            raise ValueError("V1_RESOLVE_REQUIRES_IP_ADDRESS")
        self.hostname, self.port, self.address = hostname, port, str(ip)
        super().__init__()

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        parsed = urlparse(request.url)
        if (parsed.scheme != "https" or parsed.hostname != self.hostname
                or (parsed.port or 443) != self.port or parsed.username is not None
                or parsed.password is not None):
            raise ValueError("V1_RESOLVE_SOURCE_ORIGIN_CHANGED")
        if proxies and any(proxies.values()):
            raise ValueError("V1_RESOLVE_CONFLICTS_WITH_PROXY_TRANSPORT")
        if verify is not True and not (isinstance(verify, str) and verify):
            raise ValueError("V1_RESOLVE_REQUIRES_CERTIFICATE_VERIFICATION")
        host, tls = self.build_connection_pool_key_attributes(request, verify, cert)
        host["host"] = self.address
        tls.update(server_hostname=self.hostname, assert_hostname=self.hostname)
        request.headers["Host"] = parsed.netloc
        return self.poolmanager.connection_from_host(**host, pool_kwargs=tls)


def _compute_storage(output: Path, required_bytes: int, reserve_bytes: int) -> dict:
    """Require Slurm compute placement and space on a resolved local mount."""
    node = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") != node:
        raise ValueError("V1_ASSET_BODY_REQUIRES_COMPUTE_ALLOCATION")
    ancestor = output.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    filesystem = subprocess.run(
        ["findmnt", "-T", str(ancestor), "-n", "-o", "FSTYPE"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if filesystem not in LOCAL_FILESYSTEMS:
        raise ValueError("V1_BULK_DESTINATION_REQUIRES_NODE_LOCAL_STORAGE")
    free = shutil.disk_usage(ancestor).free
    if required_bytes + reserve_bytes > free:
        raise ValueError("V1_BULK_DESTINATION_DISK_ALLOWANCE_EXCEEDED")
    return dict(filesystem=filesystem, free_bytes_before=free,
                required_bytes=required_bytes, disk_reserve_bytes=reserve_bytes)


def _range_response(response, source: dict, start: int, stop: int, *, ranged: bool, metadata: bool):
    response.raise_for_status()
    if (source.get("etag") and (not ranged or _strong_etag(source))
            and response.headers.get("ETag") != source["etag"]):
        raise ValueError("V1_ASSET_ETAG_CHANGED")
    if ranged and (response.status_code != 206 or response.headers.get("Content-Range")
                   != f"bytes {start}-{stop}/{source['expected_bytes']}"):
        raise ValueError("V1_METADATA_RANGE_MISMATCH" if metadata else "V1_ASSET_RANGE_MISMATCH")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("V1_ASSET_CONTENT_ENCODING_CHANGED")


def _resume_prefix(directory: Path, source: dict, *, metadata: bool, chunk_size: int) -> tuple[Path, int, str | None]:
    prior = json.loads((directory / "source.json").read_text())
    partial = directory / (source["artifact"] + ".part")
    actual = partial.stat().st_size
    count = prior.get("verified_prefix_bytes")
    expected_hash = prior.get("verified_prefix_sha256")
    # Retain compatibility with the already recorded metadata timeout receipt.
    legacy = (metadata and count is None and prior.get("error") == "V1_METADATA_TRANSFER_TIME_LIMIT"
              and actual % chunk_size == 0)
    if legacy:
        count = actual
    if (prior.get("source") != source or actual != prior.get("received_bytes")
            or not isinstance(count, int) or count < 0 or count > actual
            or count >= int(source["expected_bytes"])
            or (not legacy and (not prior.get("ranged_transfer") or not expected_hash))):
        raise ValueError("V1_VERIFIED_METADATA_PREFIX_REQUIRED" if metadata else "V1_VERIFIED_ASSET_PREFIX_REQUIRED")
    return partial, count, expected_hash


def download_resolved(
    receipt: Path, output: Path, *, metadata_transfer=False,
    resume_metadata: Path | None = None, asset_trust_env=False,
    range_bytes: int | None = None, resume_from: Path | None = None,
    max_transfer_bytes: int | None = None, attempt_seconds: float | None = None,
    disk_reserve_bytes: int = 0, probe_bytes: int | None = None,
    resolve_address: str | None = None, range_retries: int = 0, use_resolved_url: bool = False,
):
    """Acquire an exact source asset, optionally using resumable proxied ranges.

    Bulk bodies require a compute step and node-local space. Metadata remains
    capped at 16 MiB/240 seconds. A byte-limited attempt preserves its verified
    prefix; a probe records transport evidence without declaring asset intake.
    A source-declared whole-file checksum can bind ranges without a strong ETag.
    An explicit address routes only this session and retains HTTPS verification.
    Optional retries restart only an interrupted range; identity failures remain fatal.
    """
    source = json.loads(receipt.read_text())
    name, size = source["artifact"], int(source["expected_bytes"])
    if Path(name).name != name or name in {".", ".."} or size <= 0:
        raise ValueError("INVALID_V1_DOWNLOAD_IDENTITY")
    if metadata_transfer and size > METADATA_BYTE_LIMIT:
        raise ValueError("V1_METADATA_TRANSFER_EXCEEDS_SMALL_FILE_LIMIT")
    if resume_metadata is not None and (not metadata_transfer or resume_from is not None):
        raise ValueError("V1_RESUME_METADATA_REQUIRES_METADATA_TRANSFER")
    if any(value is not None and (not isinstance(value, int) or value <= 0)
           for value in (range_bytes, max_transfer_bytes, probe_bytes)):
        raise ValueError("V1_TRANSFER_BYTE_LIMIT_MUST_BE_POSITIVE_INTEGER")
    if not isinstance(disk_reserve_bytes, int) or disk_reserve_bytes < 0:
        raise ValueError("V1_DISK_RESERVE_MUST_BE_NONNEGATIVE_INTEGER")
    if attempt_seconds is not None and (not math.isfinite(attempt_seconds) or attempt_seconds <= 0):
        raise ValueError("V1_ATTEMPT_SECONDS_MUST_BE_POSITIVE_FINITE")
    if metadata_transfer and ((range_bytes is not None and range_bytes > DEFAULT_RANGE_BYTES)
                              or (attempt_seconds is not None and attempt_seconds > METADATA_ATTEMPT_SECONDS)):
        raise ValueError("V1_METADATA_TRANSFER_BOUND_CANNOT_BE_INCREASED")
    resume = resume_metadata or resume_from
    if probe_bytes is not None and (resume is not None or max_transfer_bytes is not None):
        raise ValueError("V1_PROBE_CANNOT_RESUME_OR_SET_SECOND_BYTE_LIMIT")
    ranged = bool(metadata_transfer or asset_trust_env or range_bytes is not None
                  or resume is not None or max_transfer_bytes is not None or probe_bytes is not None)
    if type(range_retries) is not int or not 0 <= range_retries <= 2:
        raise ValueError("V1_RANGE_RETRIES_MUST_BE_ZERO_ONE_OR_TWO")
    if range_retries and not ranged:
        raise ValueError("V1_RETRIES_REQUIRE_RANGED_TRANSFER")
    checksums = _expected_checksums(source)
    if ranged and not _strong_etag(source) and not checksums:
        raise ValueError("V1_RANGED_TRANSFER_REQUIRES_STRONG_ETAG_OR_SOURCE_CHECKSUM")
    # A resolved CDN origin can be selected explicitly when the published URL
    # redirects across origins. The IP adapter still authenticates that origin,
    # and the existing ETag/range checks bind every response to the same file.
    download_url = (source["published_url"].replace("&dl=0", "&dl=1")
                    if ranged and not use_resolved_url else source["resolved_url"])
    if resolve_address is not None and asset_trust_env:
        raise ValueError("V1_RESOLVE_CONFLICTS_WITH_ENVIRONMENT_TRANSPORT")
    route = _ResolvedAddressAdapter(download_url, resolve_address) if resolve_address is not None else None
    chunk_size = (range_bytes or DEFAULT_RANGE_BYTES) if ranged else size
    seconds = (attempt_seconds or METADATA_ATTEMPT_SECONDS) if metadata_transfer else attempt_seconds
    storage = None
    if not metadata_transfer:
        storage = _compute_storage(output, min(size, probe_bytes) if probe_bytes else size, disk_reserve_bytes)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source=source, status="DOWNLOADING", asset_body_acquired=False,
                  complete_model_acquired=False, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  slurm_step_id=os.environ.get("SLURM_STEP_ID"), metadata_transfer=metadata_transfer,
                  asset_transport=("explicit_address" if route is not None else
                                   "configured_environment" if metadata_transfer or asset_trust_env else "direct"),
                  resolved_address_route=(dict(hostname=route.hostname, address=route.address, port=route.port,
                                               tls_hostname=route.hostname, certificate_verification=True)
                                          if route is not None else None),
                  range_identity=("strong_etag" if _strong_etag(source) else "source_checksum") if ranged else None,
                  checksum_verification={algorithm: dict(expected=value, actual=None, verified=False)
                                         for algorithm, value in checksums.items()},
                  requested_url_source="resolved_url" if use_resolved_url or not ranged else "published_url",
                  ranged_transfer=ranged, range_bytes=chunk_size if ranged else None,
                  max_transfer_bytes=max_transfer_bytes, attempt_seconds=seconds,
                  probe_bytes=probe_bytes, storage=storage, range_retries=range_retries,
                  range_transfer_failures=[])
    session = requests.Session()
    session.trust_env = (metadata_transfer or asset_trust_env) and route is None
    if route is not None:
        session.mount("https://", route)
        session.mount("http://", route)
        session.headers["User-Agent"] = "NeuronSim-source-check/1.0"
    digest, actual, verified = hashlib.sha256(), 0, 0
    md5 = hashlib.md5(usedforsecurity=False) if "md5" in checksums else None
    verified_hash = digest.hexdigest()
    deadline = time.monotonic() + seconds if seconds is not None else None
    timeout_error = "V1_METADATA_TRANSFER_TIME_LIMIT" if metadata_transfer else "V1_ASSET_TRANSFER_TIME_LIMIT"
    try:
        target = output / name
        partial = output / (name + ".part")
        # A prior archive may be on shared storage: stage only its verified
        # prefix into this node-local output before requesting new ranges.
        # The archive stays read-only; its incomplete range tail is not reused.
        # A checksum-bound prefix remains provisional until the whole file matches.
        with partial.open("xb") as stream:
            if resume is not None:
                prior_partial, prefix_bytes, expected_hash = _resume_prefix(
                    resume, source, metadata=metadata_transfer, chunk_size=chunk_size)
                with prior_partial.open("rb") as previous:
                    while actual < prefix_bytes:
                        block = previous.read(min(1024 * 1024, prefix_bytes - actual))
                        if not block:
                            raise ValueError("V1_RESUME_PREFIX_TRUNCATED")
                        digest.update(block)
                        if md5 is not None:
                            md5.update(block)
                        stream.write(block)
                        actual += len(block)
                if expected_hash is not None and digest.hexdigest() != expected_hash:
                    raise ValueError("V1_RESUME_PREFIX_HASH_CHANGED")
                verified, verified_hash = actual, digest.hexdigest()
                report.update(resumed_from=str(resume), resumed_prefix_bytes=actual,
                              resumed_prefix_sha256=verified_hash)
            initial = actual
            limit = min(size, probe_bytes) if probe_bytes is not None else size
            if max_transfer_bytes is not None:
                limit = min(limit, initial + max_transfer_bytes)
            for start in range(initial, limit, chunk_size):
                stop = min(limit, start + chunk_size) - 1
                headers = {"Accept-Encoding": "identity"}
                if ranged:
                    headers["Range"] = f"bytes={start}-{stop}"
                    if _strong_etag(source):
                        headers["If-Match"] = source["etag"]
                before_digest = digest.copy()
                before_md5 = md5.copy() if md5 is not None else None
                for attempt in range(range_retries + 1):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(timeout_error)
                    read_timeout = 45 if metadata_transfer else 90
                    if deadline is not None:
                        read_timeout = min(read_timeout, max(0.001, deadline - time.monotonic()))
                    try:
                        with session.get(download_url, headers=headers, stream=True,
                                         timeout=(min(15, read_timeout), read_timeout)) as response:
                            _range_response(response, source, start, stop, ranged=ranged, metadata=metadata_transfer)
                            report["resolved_url"] = response.url
                            received = 0
                            for chunk in response.iter_content(65536):
                                if deadline is not None and time.monotonic() >= deadline:
                                    raise TimeoutError(timeout_error)
                                if received + len(chunk) > stop - start + 1:
                                    raise ValueError("V1_ASSET_SIZE_EXCEEDED")
                                received += len(chunk)
                                digest.update(chunk)
                                if md5 is not None:
                                    md5.update(chunk)
                                stream.write(chunk)
                                actual += len(chunk)
                            if received != stop - start + 1:
                                raise ValueError("V1_ASSET_INCOMPLETE")
                            verified, verified_hash = actual, digest.hexdigest()
                        break
                    except requests.RequestException as error:
                        http_status = error.response.status_code if error.response is not None else None
                        retryable = (isinstance(error, (requests.exceptions.ChunkedEncodingError,
                                                        requests.exceptions.ReadTimeout))
                                     or isinstance(error, requests.HTTPError) and http_status == 502)
                        retry = retryable and attempt < range_retries
                        delay = float(attempt + 1)
                        if retry and deadline is not None and time.monotonic() + delay >= deadline:
                            retry = False
                        failure = dict(
                            start=start, stop=stop, failed_attempt=attempt + 1,
                            error_type=type(error).__name__, http_status=http_status,
                            unverified_bytes=actual - start, discarded_bytes=0, retry_scheduled=retry)
                        report["range_transfer_failures"].append(failure)
                        if not retry:
                            raise
                        # Only this attempt's unverified tail is discarded. Copies
                        # retain the digest state even across a second failed retry.
                        stream.seek(start)
                        stream.truncate()
                        failure["discarded_bytes"] = actual - start
                        actual = start
                        digest = before_digest.copy()
                        md5 = before_md5.copy() if before_md5 is not None else None
                        time.sleep(delay)
            if probe_bytes is not None:
                report.update(status="SOURCE_RANGE_PROBED", probed_bytes=actual,
                              sha256=digest.hexdigest())
            elif actual < size:
                report.update(status="PARTIAL", stop_reason="V1_ASSET_TRANSFER_BYTE_LIMIT")
            else:
                actual_checksums = {"sha256": digest.hexdigest()}
                if md5 is not None:
                    actual_checksums["md5"] = md5.hexdigest()
                report["checksum_verification"] = {
                    algorithm: dict(expected=expected, actual=actual_checksums[algorithm],
                                    verified=actual_checksums[algorithm] == expected)
                    for algorithm, expected in checksums.items()
                }
                if "sha256" in checksums and actual_checksums["sha256"] != checksums["sha256"]:
                    raise ValueError("V1_ASSET_HASH_CHANGED")
                if "md5" in checksums and actual_checksums["md5"] != checksums["md5"]:
                    raise ValueError("V1_ASSET_MD5_CHANGED")
                # An exclusive hard link prevents replacing an unexpected target.
                stream.flush()
                os.link(partial, target)
                partial.unlink()
                report.update(status="SOURCE_ASSET_ACQUIRED", asset_body_acquired=True,
                              artifact=name, bytes=actual, sha256=digest.hexdigest())
    except KeyboardInterrupt:
        report.update(status="FAILED", error_type="KeyboardInterrupt", error="V1_SOURCE_INTERRUPTED")
        raise
    except requests.RequestException as error:
        report.update(status="FAILED", error_type=type(error).__name__, error="V1_ASSET_TRANSPORT_FAILED")
        if error.response is not None:
            report["http_status"] = error.response.status_code
        # Request exceptions can embed credential-bearing proxy URLs.
        raise RuntimeError("V1_ASSET_TRANSPORT_FAILED:" + type(error).__name__) from None
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(received_bytes=actual, verified_prefix_bytes=verified,
                      verified_prefix_sha256=verified_hash)
        session.close()
        freeze_json(output / "source.json", report)
    return report


def discover(output: Path, *, asset_trust_env=False):
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    asset_session = requests.Session()
    asset_session.trust_env = asset_trust_env
    report = dict(status="SOURCE_PAGE_PENDING", model_page=MODEL_PAGE,
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  asset_transport="configured_environment" if asset_trust_env else "direct",
                  model_assets_acquired=False, native_model_qualified=False)
    try:
        response = session.get(MODEL_PAGE, timeout=(15, 45))
        response.raise_for_status()
        page = response.content
        (output / "allen_model_page.html").write_bytes(page)
        links = sorted(set(html.unescape(url) for url in re.findall(
            r'https?[^\s<>"\\]*dropbox[^\s<>"\\]*', page.decode("utf-8"))))
        report.update(model_page_sha256=hashlib.sha256(page).hexdigest(), published_asset_links=links)
        if not links:
            raise ValueError("ALLEN_PAGE_MODEL_ASSET_LINK_MISSING")
        indexes = []
        for number, link in enumerate(links):
            with asset_session.get(link, stream=True, timeout=(15,45)) as source:
                source.raise_for_status()
                entry = dict(published_url=link, resolved_url=source.url,
                             content_type=source.headers.get("Content-Type"),
                             declared_bytes=source.headers.get("Content-Length"))
                if "html" not in source.headers.get("Content-Type", "").lower():
                    entry["status"] = "DIRECT_ASSET_REQUIRES_SIZE_AND_VERSION_QUALIFICATION"
                else:
                    pieces, size = [], 0
                    for chunk in source.iter_content(65536):
                        size += len(chunk)
                        if size > 8*1024*1024:
                            raise ValueError("MODEL_ASSET_INDEX_EXCEEDS_METADATA_BOUND")
                        pieces.append(chunk)
                    content = b"".join(pieces)
                    (output / f"asset_index_{number}.html").write_bytes(content)
                    entry.update(status="INDEX_FETCHED", sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
                    entry["archive_name_candidates"] = sorted(set(re.findall(
                        r'[^"<>\\\n]{1,150}\.zip', content.decode("utf-8"))))
                    entry["observed_child_urls"] = folder_urls(content.decode("utf-8"))
                indexes.append(entry)
        report.update(status="ASSET_INDEX_RESOLVED_MODEL_FILES_UNACQUIRED", asset_indexes=indexes)
    except (requests.RequestException, ValueError) as error:
        report.update(status="BLOCKED_ASSET_INTAKE", error_type=type(error).__name__,
                      error="Configured asset transport failed" if asset_trust_env else str(error))
    finally:
        session.close()
        asset_session.close()
        freeze_json(output / "asset_intake.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-trust-env", action="store_true", help="Use configured environment transport for assets; bulk uses exact ranges")
    parser.add_argument("--parent-index", type=Path)
    parser.add_argument("--child-name")
    parser.add_argument("--text-asset", action="store_true")
    parser.add_argument("--resolve-download", action="store_true")
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--metadata-transfer", action="store_true")
    parser.add_argument("--resume-metadata", type=Path)
    parser.add_argument("--range-bytes", type=int, help="Exact bounded ranges; bulk default remains a direct stream")
    parser.add_argument("--resume-from", type=Path, help="Prior output containing a verified partial prefix")
    parser.add_argument("--range-retries", type=int, default=0, help="Zero to two retries of a transiently failed range")
    parser.add_argument("--max-transfer-bytes", type=int, help="Maximum new body bytes in this attempt")
    parser.add_argument("--attempt-seconds", type=float, help="Optional attempt ceiling; bulk has none by default")
    parser.add_argument("--disk-reserve-bytes", type=int, default=0)
    parser.add_argument("--probe-bytes", type=int, help="Probe this many source bytes without acquiring the asset")
    parser.add_argument("--resolve-address", help="Route this source HTTPS hostname to an explicit IP, retaining TLS verification")
    parser.add_argument("--use-resolved-url", action="store_true", help="Start ranged requests at the receipt-bound resolved origin; preserve ETag, ranges and TLS checks")
    args = parser.parse_args()
    if args.source_receipt is not None:
        result = download_resolved(
            args.source_receipt, args.output, metadata_transfer=args.metadata_transfer,
            resume_metadata=args.resume_metadata, asset_trust_env=args.asset_trust_env,
            range_bytes=args.range_bytes, resume_from=args.resume_from,
            max_transfer_bytes=args.max_transfer_bytes, attempt_seconds=args.attempt_seconds,
            disk_reserve_bytes=args.disk_reserve_bytes, probe_bytes=args.probe_bytes,
            resolve_address=args.resolve_address, range_retries=args.range_retries,
            use_resolved_url=args.use_resolved_url,
        )
        print(json.dumps(result))
        if result["status"] == "PARTIAL":
            raise SystemExit(2)
        return
    if (args.metadata_transfer or args.resume_metadata is not None or args.range_bytes is not None
            or args.resume_from is not None or args.max_transfer_bytes is not None
            or args.attempt_seconds is not None or args.disk_reserve_bytes or args.probe_bytes is not None
            or args.resolve_address is not None or args.range_retries or args.use_resolved_url):
        parser.error("download transport options require --source-receipt")
    if args.parent_index is not None:
        if args.child_name is None:
            parser.error("--parent-index requires --child-name")
        if args.resolve_download:
            print(json.dumps(resolve_download(args.parent_index, args.child_name, args.output)))
        else:
            print(json.dumps(fetch_child(args.parent_index, args.child_name, args.output, text_asset=args.text_asset)))
        return
    result = discover(args.output, asset_trust_env=args.asset_trust_env)
    print(json.dumps(result))
    if result["status"] == "BLOCKED_ASSET_INTAKE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

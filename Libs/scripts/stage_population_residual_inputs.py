"""Transfer manifest-bound native fitting inputs in bounded serial batches.

This performs byte transfer only. An uncertain upload stops before replay; the
pending record identifies the exact name, size and hash for remote reconciliation.
--start-file permits independent later files while retaining an unresolved record.
"""
from __future__ import annotations

import argparse
import base64
import shutil
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import tempfile
import time


DATA_CONTROL_PLANE = 'cn-seed'
DATA_UPLOAD_ENDPOINT = 'https://cluster-seed.example.invalid/api/v4/seed/data/datacard-upload-files/create'


def upload_once(api, data, name):
    """Use the CLI JSON upload contract with one attempt and stdin-only credentials."""
    session = api.http_session()
    headers = ['x-jwt-token: ' + session.headers['x-jwt-token'], 'Accept-Encoding: identity', 'Content-Type: application/json']
    cookies = '; '.join(k + '=' + v for k, v in session.cookies.get_dict().items())
    if cookies:
        headers.append('Cookie: ' + cookies)
    configuration = '\n'.join('header = ' + json.dumps(v) for v in headers) + '\n'
    with tempfile.NamedTemporaryFile(prefix='neuronsim-native-input-', suffix='.part') as temporary:
        temporary.write(json.dumps({'file_name': name, 'content_base64': base64.b64encode(data).decode()}).encode())
        temporary.flush()
        result = subprocess.run(['curl', '--silent', '--show-error', '--connect-timeout', '15',
            '--max-time', '180', '--proxy', session.proxies['https'],
            '--config', '-', '--data-binary', '@' + temporary.name,
            '--write-out', '\n{"http_code":"%{http_code}","size_upload":%{size_upload},"time_appconnect":%{time_appconnect},"time_total":%{time_total}}',
            DATA_UPLOAD_ENDPOINT],
            input=configuration, capture_output=True, text=True)
    body, metadata = result.stdout.rsplit('\n', 1)
    metadata = json.loads(metadata)
    return result.returncode, metadata, body, result.stderr[-1000:]


def upload_cli_once(data, name):
    cli = os.environ.get('MERLIN_CLI') or shutil.which('merlin-cli')
    if cli is None:
        raise FileNotFoundError('authenticated merlin-cli required')
    with tempfile.NamedTemporaryFile(mode='w', prefix='neuronsim-native-input-', suffix='.json') as request:
        json.dump({'file_name': name, 'content_base64': base64.b64encode(data).decode()}, request)
        request.flush()
        result = subprocess.run([cli, '--control-plane', DATA_CONTROL_PLANE, 'data', 'datacard-upload-files',
            'create', '--from-file', request.name], cwd=Path(request.name).parent,
            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Merlin CLI upload failed: ' + result.stderr[-1500:])
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--start-file", type=int, default=0)
    parser.add_argument("--transport", choices=("curl", "cli"), default="curl")
    parser.add_argument("--skip-pending", action="store_true", help="Continue independent files while retaining unknown byte identities")
    parser.add_argument("--reuse-resolved", type=Path, help="Reuse authenticated input locations by byte identity")
    args = parser.parse_args()
    started = time.monotonic()
    if not 1 <= args.max_files <= 1000:
        raise ValueError("one to1000 serial uploads per invocation required")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "native_residual_observation_inputs_v1"
            or manifest.get("training_seeds") != [41, 42, 43]
            or manifest.get("source_units") != 100000):
        raise ValueError("registered native residual input manifest required")
    files = manifest["files"]
    if len({row["path"] for row in files}) != len(files):
        raise ValueError("duplicate input paths")
    hostname = platform.node().split(".")[0]
    if hostname == "mgmt2":
        if manifest["bytes"] > 256 * 1024**2 or len(files) > 1000 or args.max_files > 2:
            raise RuntimeError("bulk fitting transfer requires a source-side I/O host")
    elif hostname not in ("cpu03", "cpu04"):
        raise RuntimeError("authorized source-side I/O host required")
    if any(not 0 < row["bytes"] <= 64 * 1024**2 for row in files):
        raise ValueError("each file must fit the existing64MiB upload limit")
    os.environ.setdefault('MERLIN_AUTH_FILE', str(Path.home() / '.merlin-cli/auth/cn-seed.json'))
    sys.path.insert(0, str(args.helper_dir))
    import merlin_io
    args.output.mkdir(parents=True, exist_ok=True)
    state_path, pending_path = args.output / "upload.json", args.output / "pending.json"
    digest = hashlib.sha256(raw).hexdigest()
    if not 0 <= args.start_file < len(files):
        raise ValueError("start-file must select a manifest row")
    if pending_path.exists():
        legacy = json.loads(pending_path.read_text())
        index = next(i for i, row in enumerate(files) if row['path'] == legacy['path'])
        target = args.output / f'pending_{index:03d}.json'
        if target.exists():
            raise RuntimeError("conflicting pending upload records")
        pending_path.rename(target)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        manifest_sha256=digest, files=[], complete=False, source_host=hostname,
        scientific_computation=False, scientific_backend="merlin")
    if state["manifest_sha256"] != digest:
        raise ValueError("input manifest changed during transfer")
    transferred = {row["path"]: row for row in state["files"]}
    if len(transferred) != len(state["files"]):
        raise ValueError("duplicate returned input paths")
    for row in state["files"]:
        expected = next((x for x in files if x["path"] == row["path"]), None)
        if expected is None or any(row[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("returned upload differs from input identity")
    reusable = {(row['sha256'], row['bytes']): row for row in state['files']}
    if args.reuse_resolved:
        prior = json.loads(args.reuse_resolved.read_text())
        if prior.get('complete') is not True:
            raise ValueError('complete authenticated prior input resolution required')
        for row in prior['files']:
            if 'hdfs_path' in row or row.get('resolution') == 'inline_original_metadata':
                reusable.setdefault((row['sha256'], row['bytes']), row)
    pending_identities = {}
    for marker in args.output.glob("pending_*.json"):
        unresolved = json.loads(marker.read_text())
        if unresolved.get("status") == "CREATE_OUTCOME_UNKNOWN":
            pending_identities[(unresolved["sha256"], unresolved["bytes"])] = unresolved["path"]
    done = 0
    for index, row in enumerate(files):
        if index < args.start_file or row["path"] in transferred:
            continue
        identity = (row['sha256'], row['bytes'])
        if identity in pending_identities and (args.skip_pending or row["path"] != pending_identities[identity]):
            # A different logical name does not authorize replay of unknown bytes.
            continue
        inline_metadata = row['path'].endswith('.json') and row['bytes'] <= 64 * 1024
        if identity in reusable or inline_metadata:
            prior = reusable.get(identity, row)
            reused = {key: row[key] for key in ('path', 'bytes', 'sha256')}
            if identity in reusable:
                reused['reused_from'] = prior['path']
            if 'hdfs_path' in prior:
                reused['hdfs_path'] = prior['hdfs_path']
            else:
                data = Path(row['source']).read_bytes()
                if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise ValueError('inline input identity differs')
                reused.update(content_base64=base64.b64encode(data).decode(),
                              resolution='inline_original_metadata')
            state['files'].append(reused)
            reusable[identity] = reused
            state.update(complete=len(state['files']) == len(files),
                         transferred_bytes=sum(x['bytes'] for x in state['files']),
                         expected_bytes=manifest['bytes'])
            merlin_io.atomic_json(state_path, state)
            continue
        pending_path = args.output / f'pending_{index:03d}.json'
        if pending_path.exists():
            raise RuntimeError("reconcile " + pending_path.name + " before replaying this file")
        data = Path(row["source"]).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("source input changed: " + row["path"])
        name = f"nrnsim_native_fit_{index:03d}_{row['sha256'][:16]}.bin"
        pending = dict(path=row["path"], file_name=name, bytes=len(data), sha256=row["sha256"],
            status="CREATE_DISPATCHED", dispatched_at=datetime.now(timezone.utc).isoformat())
        merlin_io.atomic_json(pending_path, pending)
        try:
            if args.transport == "cli":
                pending["transport"] = "merlin_cli_json_local_cwd"
                merlin_io.atomic_json(pending_path, pending)
                value = upload_cli_once(data, name)
            else:
                code, transport, body, error = upload_once(merlin_io, data, name)
                http_status = int(transport["http_code"])
                pending["transport"] = transport
                if code and transport["size_upload"] == 0 and transport["time_appconnect"] == 0:
                    pending.update(status="NOT_SENT_CONNECTION_FAILED", error=error)
                    merlin_io.atomic_json(args.output / f"connection_failure_{index:03d}.json", pending)
                    pending_path.unlink()
                    raise RuntimeError("upload connection failed before TLS and body transmission: " + error)
                if 400 <= http_status < 500:
                    pending.update(status="CONFIRMED_HTTP_REJECTION", http_status=http_status)
                    merlin_io.atomic_json(args.output / f"rejected_{index:03d}.json", pending)
                    pending_path.unlink()
                    raise RuntimeError("upload rejected with HTTP" + str(http_status))
                if code or not 200 <= http_status < 300:
                    raise RuntimeError(f"upload transport exit={code} HTTP={http_status}: {error}")
                value = json.loads(body)
            if value.get("ResponseBase", {}).get("code", 0) != 0:
                raise RuntimeError("upload API returned an error; reconcile outcome")
            accepted = value.get("data", value)
            pending["accepted"] = {key: accepted[key] for key in ("hdfs_path", "size_bytes")}
            merlin_io.atomic_json(pending_path, pending)
            if accepted["size_bytes"] != len(data):
                raise ValueError("accepted file size differs; reconcile outcome")
            state["files"].append(dict(path=row["path"], bytes=len(data), sha256=row["sha256"],
                hdfs_path=accepted["hdfs_path"], uploaded_at=datetime.now(timezone.utc).isoformat()))
            reusable[identity] = state['files'][-1]
            state.update(complete=len(state["files"]) == len(files),
                transferred_bytes=sum(x["bytes"] for x in state["files"]), expected_bytes=manifest["bytes"])
            merlin_io.atomic_json(state_path, state)
            pending_path.unlink()
            done += 1
            print(json.dumps(dict(uploaded=row["path"], bytes=len(data),
                files_complete=len(state["files"]), total_files=len(files))), flush=True)
        except Exception as error:
            if pending_path.exists():
                pending.update(status="CREATE_OUTCOME_UNKNOWN", error_type=type(error).__name__)
                merlin_io.atomic_json(pending_path, pending)
            raise
        if done == args.max_files or time.monotonic() - started > (45 if hostname == 'mgmt2' else 240):
            break
    print(json.dumps(dict(complete=state["complete"], files=len(state["files"]),
        transferred_bytes=state.get("transferred_bytes", 0))), flush=True)


if __name__ == "__main__":
    main()

"""The upload caller retains unknown dispatches and releases unsent requests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import socketserver
import sys
import threading
from types import SimpleNamespace

import pytest
import requests


def uploader_fixture(tmp_path, monkeypatch, session):
    spec = importlib.util.spec_from_file_location(
        "pfc_original_upload", Path(__file__).with_name("upload_pfc_original_events.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "source.npz"
    raw.write_bytes(b"deterministic original-event transport fixture")
    inputs = tmp_path / "inputs.json"
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    inputs.write_text(json.dumps([
        dict(index=i, path=str(raw), bytes=raw.stat().st_size, sha256=digest,
             identity={}, source_report=str(tmp_path / "source_report.json"))
        for i in range(35)]))
    output = tmp_path / "uploads"
    monkeypatch.setattr(module, "socket", SimpleNamespace(gethostname=lambda: "cpu05"))
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=lambda _: None))
    real_path = Path
    monkeypatch.setattr(module, "Path", lambda value: (
        tmp_path / "scratch" if str(value) == "/tmp/neuronsim-pfc-original35-transfer"
        else real_path(value)))
    monkeypatch.setattr(module, "http_session", lambda: session)
    monkeypatch.setattr(sys, "argv", [
        "upload", "--inputs", str(inputs), "--output", str(output),
        "--start", "0", "--stop", "1", "--chunk-mib", "1"])
    return module, output


def test_proxy_connect_reset_does_not_leave_unknown_upload(tmp_path, monkeypatch):
    exchanges = []

    class CloseConnect(socketserver.StreamRequestHandler):
        def handle(self):
            headers = []
            while True:
                line = self.rfile.readline()
                headers.append(line)
                if line in (b"", b"\r\n"):
                    break
            exchanges.append(b"".join(headers))
            self.request.shutdown(socket.SHUT_RDWR)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), CloseConnect) as proxy:
        proxy.daemon_threads = True
        worker = threading.Thread(target=proxy.serve_forever, daemon=True)
        worker.start()
        session = requests.Session()
        session.trust_env = False
        session.proxies.update(https=f"http://127.0.0.1:{proxy.server_address[1]}")
        module, output = uploader_fixture(tmp_path, monkeypatch, session)
        try:
            with pytest.raises(module.UploadRequestNotSentError):
                module.main()
        finally:
            session.close()
            proxy.shutdown()
            worker.join()
    assert len(exchanges) == 3
    assert all(value.startswith(b"CONNECT ") for value in exchanges)
    assert not (output / "task_0000_inflight.json").exists()
    state = json.loads((output / "task_0000_upload.json").read_text())
    failure = json.loads((output / "task_0000_proxy_failure.json").read_text())
    assert state["chunks"] == [] and state["complete"] is False
    assert failure["status"] == "NOT_SENT_PROXY_CONNECTION_FAILED"


def test_read_timeout_after_dispatch_keeps_unknown_upload(tmp_path, monkeypatch):
    calls = []

    class DispatchedSession:
        def post(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            calls.append(kwargs["files"]["file"][1].getvalue())
            raise requests.exceptions.ReadTimeout("Response missing after dispatch")

    module, output = uploader_fixture(tmp_path, monkeypatch, DispatchedSession())
    with pytest.raises(requests.exceptions.ReadTimeout):
        module.main()
    assert len(calls) == 1
    pending = json.loads((output / "task_0000_inflight.json").read_text())
    state = json.loads((output / "task_0000_upload.json").read_text())
    assert pending["status"] == "CREATE_OUTCOME_UNKNOWN"
    assert pending["error_type"] == "ReadTimeout"
    assert state["chunks"] == [] and state["complete"] is False
    assert not (output / "task_0000_proxy_failure.json").exists()

def test_redirect_response_keeps_unknown_upload(tmp_path, monkeypatch):
    calls = []

    class RedirectSession:
        def post(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            calls.append(kwargs["files"]["file"][1].getvalue())
            return SimpleNamespace(status_code=307)

    module, output = uploader_fixture(tmp_path, monkeypatch, RedirectSession())
    with pytest.raises(RuntimeError, match="REDIRECT_OUTCOME_UNKNOWN"):
        module.main()
    assert len(calls) == 1
    pending = json.loads((output / "task_0000_inflight.json").read_text())
    state = json.loads((output / "task_0000_upload.json").read_text())
    assert pending["status"] == "CREATE_OUTCOME_UNKNOWN"
    assert state["chunks"] == [] and state["complete"] is False
    assert not (output / "task_0000_proxy_failure.json").exists()

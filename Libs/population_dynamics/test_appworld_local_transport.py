"""Real subprocess framing and unresolved-request semantics for the model host."""
import json
from pathlib import Path
import sys

import pytest

from Libs.scripts.run_appworld_local_collaboration import ProcessJsonChannel


def test_buffered_utf8_finish_and_terminal_are_read_without_waiting_for_eof(tmp_path):
    program = """
import json,os,sys,time
request=json.loads(sys.stdin.readline())
payload=(json.dumps({'ok':True,'result':request},ensure_ascii=False)+'\\n'
         +json.dumps({'terminal':{'status':'DEVELOPMENT_EXECUTION_COMPLETE'}})+'\\n').encode()
split=payload.index('视图'.encode())+1
os.write(1,payload[:split])
time.sleep(0.02)
os.write(1,payload[split:])
sys.stdin.read()
"""
    channel = ProcessJsonChannel([sys.executable, "-u", "-c", program], tmp_path / "stderr", 5)
    try:
        request = {"actor": "executor", "text": "完整视图"}
        channel.send(request)
        assert channel.receive() == {"ok": True, "result": request}
        assert channel.receive() == {"terminal": {"status": "DEVELOPMENT_EXECUTION_COMPLETE"}}
        assert channel.process.poll() is None
    finally:
        result = channel.close_and_wait()
    assert result["exit_code"] == 0 and not result["request_outcome_unresolved"]
    assert result["sent_requests"] == 1 and result["received_messages"] == 2
    assert result["late_stdout_bytes_hex"] == ""


@pytest.mark.parametrize("program,error,exit_code", [
    ("import sys;sys.stdin.readline();print('not json',flush=True)", ValueError, 0),
    ("import sys;sys.stdin.readline();sys.exit(7)", EOFError, 7),
])
def test_bad_ack_does_not_permit_second_send(tmp_path, program, error, exit_code):
    channel = ProcessJsonChannel([sys.executable, "-u", "-c", program], tmp_path / "stderr", 5)
    request = {"kind": "tool", "call_id": "one_write"}
    try:
        channel.send(request)
        with pytest.raises(error):
            channel.receive()
        with pytest.raises(RuntimeError, match="ALREADY_PENDING"):
            channel.send(request)
    finally:
        result = channel.close_and_wait()
    assert result["exit_code"] == exit_code
    assert result["sent_requests"] == 1 and result["received_messages"] == 0
    assert result["request_outcome_unresolved"] is True and result["last_pending_request"] == request


def test_missing_ack_stays_unknown_after_clean_process_exit(tmp_path):
    program = "import sys;sys.stdin.readline();sys.stdin.read()"
    channel = ProcessJsonChannel([sys.executable, "-u", "-c", program], tmp_path / "stderr", 0.1)
    try:
        channel.send({"kind": "finish"})
        with pytest.raises(TimeoutError, match="UNRESOLVED"):
            channel.receive()
    finally:
        result = channel.close_and_wait()
    assert result["exit_code"] == 0 and result["request_outcome_unresolved"] is True
    assert result["sent_requests"] == 1 and not result["termination_signals"]

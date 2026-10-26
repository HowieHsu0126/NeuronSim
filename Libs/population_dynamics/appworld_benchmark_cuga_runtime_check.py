"""Check installed CUGA interfaces using one synthetic HTTP completion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .appworld_benchmark_cuga_process import _verify_expected_packages, _verify_worker_runtime


def mark(report, name, status):
    report["stage"] = name
    report["stages"][name] = status


def verify_identity(config, report):
    _verify_worker_runtime(
        config["python_runtime"], report.setdefault("python_runtime", {})
    )
    report["package_identity"] = {}
    _verify_expected_packages(config["expected_packages"], report["package_identity"])


async def check(args, report, requests, base_url):
    mark(report, "import", "running")
    with args.worker_config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    verify_identity(config, report)
    if config["seed"] != args.seed or config["model_name"] != args.model_name:
        raise ValueError("Diagnostic arguments differ from the worker configuration")
    from .appworld_benchmark_cuga_runtime import (
        CugaUsageLedger, load_cuga_runtime, make_usage_callback, model_client_settings,
    )
    ledger = CugaUsageLedger()
    try:
        runtime = load_cuga_runtime(
            seed=args.seed, model_name=args.model_name, base_url=base_url)
        mark(report, "import", "passed")
        mark(report, "build", "running")
        agent = runtime.DynamicAgentGraph(None, langfuse_handler=None)
        await agent.build_graph()
        report["full_graph_constructed"] = True
        report["model_client_settings"] = model_client_settings(
            runtime.manager, model_name=args.model_name, base_url=base_url, seed=args.seed)
        clients = list(runtime.manager._models.values())
        if not clients or requests:
            raise RuntimeError("Graph has no cached clients or construction requested completion")
        mark(report, "build", "passed")
        mark(report, "callback", "running")
        client = clients[0]
        report["selected_client_type"] = f"{type(client).__module__}.{type(client).__qualname__}"
        callback = make_usage_callback(ledger)
        report["callback_type"] = f"{type(callback).__module__}.{type(callback).__qualname__}"
        answer = await client.ainvoke(
            "Synthetic interface fixture. Return the fixture response.",
            config={"callbacks": [callback]},
        )
        report["sdk_usage_metadata"] = getattr(answer, "usage_metadata", None)
        report["sdk_response_metadata"] = getattr(answer, "response_metadata", None)
        if len(requests) != 1 or requests[0]["path"] != "/v1/chat/completions":
            raise RuntimeError("Expected exactly one request to /v1/chat/completions")
        body = requests[0]["json"]
        if (body.get("model") != args.model_name or body.get("seed") != args.seed
                or body.get("stream", False) is not False):
            raise RuntimeError("Actual request model, seed or streaming setting differs")
        usage = report["sdk_usage_metadata"]
        if not isinstance(usage, dict) or tuple(
            usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")
        ) != (7, 2, 9):
            raise RuntimeError("SDK usage differs from the declared fixture usage")
        summary = ledger.summary()
        if summary["recorded_usage_resolved"] is not True or summary["totals"] != {
            "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9
        }:
            raise RuntimeError("The actual callback ledger did not resolve fixture usage")
        mark(report, "callback", "passed")
    finally:
        report["ledger_calls"] = ledger.calls
        report["ledger_summary"] = ledger.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--worker-config", type=Path, required=True)
    args = parser.parse_args(argv)
    work = args.work_root.absolute()
    requests = []
    report = {
        "status": "FAIL", "stage": "setup",
        "stages": {"import": "not_run", "build": "not_run", "callback": "not_run"},
        "seed": args.seed, "synthetic_task_metadata": {"task_id": "synthetic_cuga_interface"},
        "full_graph_constructed": False, "graph_loop_executed": False,
        "synthetic_completion_requests": 0, "real_model_generations": 0,
        "native_tasks_executed": 0, "ledger_calls": None, "ledger_summary": None,
    }
    server = thread = None
    created = False
    exit_code = 1

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            entry = {"path": self.path}
            requests.append(entry)
            try:
                entry["json"] = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except Exception as error:
                entry["error"] = repr(error)
                self.send_error(400, "Invalid fixture request JSON")
                return
            response = {
                "id": "chatcmpl-cuga-fixture", "object": "chat.completion",
                "created": 0, "model": args.model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "fixture response"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
    try:
        work.mkdir(parents=True, exist_ok=False)
        created = True
        for variable, relative in {
            "CUGA_LOGGING_DIR": "logs", "CUGA_DBS_DIR": "dbs",
            "FASTEMBED_CACHE_PATH": "cache/fastembed", "XDG_CACHE_HOME": "cache",
            "HF_HOME": "cache/huggingface", "TMPDIR": "tmp",
        }.items():
            directory = work / relative
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)
        for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.pop(variable, None)
            os.environ.pop(variable.lower(), None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        os.environ["OPENAI_API_KEY"] = "synthetic-fixture-only"
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        os.environ["OPENAI_BASE_URL"] = base_url
        asyncio.run(check(args, report, requests, base_url))
        report["status"] = "PASS"
        exit_code = 0
    except BaseException as error:
        mark(report, report["stage"], "failed")
        report["exception"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
            server.server_close()
        report["synthetic_completion_requests"] = len(requests)
        report["http_requests"] = requests
        text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if created:
            (work / "cuga_runtime_check.json").write_text(text + "\n", encoding="utf-8")
        else:
            print(text, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

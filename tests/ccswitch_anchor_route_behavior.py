from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SMOKE_SECRET = "anchor-smoke-credential-not-real"
SMOKE_CONTENT = "ANCHOR_SMOKE_CONTENT_SHOULD_NOT_APPEAR"
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = int(server.server_address[1])
    server.server_close()
    return port


def get_json(url: str) -> dict[str, Any]:
    with DIRECT_OPENER.open(url, timeout=2) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{url} did not return an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise SystemExit("route binary is missing")

    capture: dict[str, Any] = {}

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0"))
            capture["path"] = self.path
            capture["authorization"] = self.headers.get("authorization")
            capture["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = json.dumps(
                {
                    "id": "resp_anchor_smoke",
                    "object": "response",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_port = int(upstream.server_address[1])
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    route_port = free_port()

    with tempfile.TemporaryDirectory(prefix="anchor-ccswitch-smoke-") as temp:
        root = Path(temp)
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        env = os.environ.copy()
        env.update(
            {
                "CC_SWITCH_TEST_HOME": str(root / "state"),
                "ANCHOR_ROUTE_ENABLED": "1",
                "ANCHOR_ROUTE_BASE_URL": f"http://127.0.0.1:{upstream_port}/v1",
                "ANCHOR_ROUTE_MODEL": "anchor-smoke-model",
                "ANCHOR_ROUTE_API_FORMAT": "openai_responses",
                "ANCHOR_ROUTE_API_KEY_ENV": "ANCHOR_ROUTE_SMOKE_KEY",
                "ANCHOR_ROUTE_SMOKE_KEY": SMOKE_SECRET,
                "ANCHOR_ROUTE_REASONING_FIELD": "reasoning.effort",
                "ANCHOR_ROUTE_REASONING_EFFORT": "max",
                "ANCHOR_ROUTE_NETWORK_MODE": "direct",
                "ANCHOR_ROUTE_LISTEN_ADDRESS": "127.0.0.1",
                "ANCHOR_ROUTE_PORT": str(route_port),
                "ANCHOR_ROUTE_MAX_RETRIES": "0",
                "ANCHOR_ROUTE_USER_AGENT": "claude-code",
            }
        )
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [str(binary)],
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        try:
            health_url = f"http://127.0.0.1:{route_port}/anchor/health"
            status_url = f"http://127.0.0.1:{route_port}/anchor/status"
            deadline = time.monotonic() + 40
            while True:
                if process.poll() is not None:
                    raise AssertionError(f"route exited before health check: {process.returncode}")
                try:
                    health = get_json(health_url)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise AssertionError("route health endpoint did not become ready")
                    time.sleep(0.25)

            request = urllib.request.Request(
                f"http://127.0.0.1:{route_port}/anchor/v1/responses",
                data=json.dumps(
                    {"model": "client-model", "input": SMOKE_CONTENT, "stream": False}
                ).encode("utf-8"),
                headers={"content-type": "application/json", "authorization": "Bearer local"},
                method="POST",
            )
            with DIRECT_OPENER.open(request, timeout=20) as response:
                if response.status != 200:
                    raise AssertionError(f"route returned HTTP {response.status}")
                json.loads(response.read().decode("utf-8"))

            status = get_json(status_url)
            if health.get("status") != "healthy" or status.get("running") is not True:
                raise AssertionError("health/status contract failed")
            if capture.get("path") != "/v1/responses":
                raise AssertionError("route forwarded to an unexpected upstream path")
            if capture.get("authorization") != f"Bearer {SMOKE_SECRET}":
                raise AssertionError("runtime environment credential was not forwarded")
            body = capture.get("body") or {}
            if body.get("reasoning") != {"effort": "max"}:
                raise AssertionError("literal reasoning.effort=max was not preserved")
            metadata = json.dumps({"health": health, "status": status}, ensure_ascii=False)
            if SMOKE_SECRET in metadata or SMOKE_CONTENT in metadata:
                raise AssertionError("content-free health/status leaked request data")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            upstream.shutdown()
            upstream.server_close()

        logs = stdout_path.read_text(encoding="utf-8", errors="replace") + stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if SMOKE_SECRET in logs or SMOKE_CONTENT in logs:
            raise AssertionError("route logs leaked the runtime credential or request content")

    print("anchor route binary behavior smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

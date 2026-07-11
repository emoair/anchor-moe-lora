from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.request import Request, urlopen

from anchor_mvp.tooling import InitialToolChoiceProxy, enforce_initial_tool_choice


def _payload(messages, *, tool_choice=None):
    value = {
        "model": "kimi-for-coding",
        "stream": True,
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read one file",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    if tool_choice is not None:
        value["tool_choice"] = tool_choice
    return value


def test_transform_forces_only_before_current_turn_tool_result():
    first, forced = enforce_initial_tool_choice(
        _payload([{"role": "user", "content": "inspect"}])
    )
    assert forced is True
    assert first["tool_choice"] == "required"

    second, forced = enforce_initial_tool_choice(
        _payload(
            [
                {"role": "user", "content": "inspect"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            tool_choice="auto",
        )
    )
    assert forced is False
    assert second["tool_choice"] == "auto"


def test_old_tool_result_does_not_satisfy_a_new_user_turn():
    transformed, forced = enforce_initial_tool_choice(
        _payload(
            [
                {"role": "user", "content": "first"},
                {"role": "tool", "tool_call_id": "old", "content": "ok"},
                {"role": "user", "content": "second"},
            ]
        )
    )
    assert forced is True
    assert transformed["tool_choice"] == "required"


def test_no_tools_is_left_unchanged():
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    transformed, forced = enforce_initial_tool_choice(payload)
    assert transformed is payload
    assert forced is False


def test_loopback_proxy_two_stage_transform_preserves_headers_and_sse_bytes():
    captures: list[tuple[dict[str, object], dict[str, str]]] = []
    sse = b'data: {"id":"one"}\n\ndata: [DONE]\n\n'

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            size = int(self.headers["Content-Length"])
            captures.append(
                (
                    json.loads(self.rfile.read(size)),
                    {name.casefold(): value for name, value in self.headers.items()},
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = (
        f"http://127.0.0.1:{upstream.server_address[1]}"
        "/coding/v1/chat/completions"
    )
    try:
        with InitialToolChoiceProxy(
            upstream_url=upstream_url, _allow_insecure_test_upstream=True
        ) as proxy:
            first = _payload([{"role": "user", "content": "inspect"}])
            first_request = Request(
                proxy.base_url + "/chat/completions",
                data=json.dumps(first).encode(),
                headers={
                    "Authorization": "Bearer test-only",
                    "Content-Type": "application/json",
                    "User-Agent": "opencode/test-real-client",
                    "X-OpenCode-Test": "preserved",
                },
                method="POST",
            )
            with urlopen(first_request, timeout=5) as response:
                assert response.read() == sse

            second = _payload(
                [
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ]
            )
            second_request = Request(
                proxy.base_url + "/chat/completions",
                data=json.dumps(second).encode(),
                headers={
                    "Authorization": "Bearer test-only",
                    "Content-Type": "application/json",
                    "User-Agent": "opencode/test-real-client",
                },
                method="POST",
            )
            with urlopen(second_request, timeout=5) as response:
                assert response.read() == sse

            assert proxy.stats.requests == 2
            assert proxy.stats.forced_requests == 1
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert captures[0][0]["tool_choice"] == "required"
    assert "tool_choice" not in captures[1][0]
    assert captures[0][1]["authorization"] == "Bearer test-only"
    assert captures[0][1]["user-agent"] == "opencode/test-real-client"
    assert captures[0][1]["x-opencode-test"] == "preserved"

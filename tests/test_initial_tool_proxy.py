from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from anchor_mvp.tooling import InitialToolChoiceProxy, enforce_initial_tool_choice
from anchor_mvp.tooling.initial_tool_proxy import _classify_kimi_400


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


def test_loopback_proxy_classifies_upstream_400_without_retaining_error_text():
    secret = "sk-private-provider-diagnostic"
    body = json.dumps(
        {
            "error": {
                "message": (
                    "thinking is enabled but reasoning_content is missing in assistant "
                    f"tool call message; {secret}"
                )
            }
        }
    ).encode()

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
            request = Request(
                proxy.base_url + "/chat/completions",
                data=json.dumps(_payload([{"role": "user", "content": "inspect"}])).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urlopen(request, timeout=5)
            except HTTPError as exc:
                assert exc.code == 400
                exc.read()
            else:
                raise AssertionError("expected upstream HTTP 400")

            stats = proxy.stats
            assert stats.error_codes == ("kimi_400_missing_reasoning_content",)
            assert secret not in repr(stats)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("(invalid_url) The provided URL is invalid", "kimi_400_invalid_url"),
        ("total message size 9 exceeds limit 2", "kimi_400_message_too_large"),
        ("Your request exceeded model token limit: 262144", "kimi_400_token_limit"),
        ("reasoning_content is missing", "kimi_400_missing_reasoning_content"),
        ("unsupported image url: local-path", "kimi_400_unsupported_image_url"),
        ("function name read is duplicated", "kimi_400_duplicate_function_name"),
        (
            "The request was rejected because it was considered high risk",
            "kimi_400_high_risk_rejected",
        ),
        ("unrecognized provider validation", "kimi_400_unknown"),
    ],
)
def test_kimi_400_classifier_returns_only_fixed_categories(message: str, expected: str):
    secret = "sk-private-classifier-input"

    code = _classify_kimi_400(f"{message}; {secret}".encode())

    assert code == expected
    assert secret not in code

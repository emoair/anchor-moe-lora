from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import ssl
import threading
from typing import Mapping
from urllib.parse import urlparse


OFFICIAL_CHAT_COMPLETIONS_URL = (
    "https://api.kimi.com/coding/v1/chat/completions"
)
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_ERROR_CLASSIFICATION_BYTES = 64 * 1024


def _classify_kimi_400(body: bytes) -> str:
    """Reduce a trusted upstream 400 body to one fixed, content-free category."""

    lowered = body.decode("utf-8", errors="replace").casefold()
    if "invalid_url" in lowered or "provided url is invalid" in lowered:
        return "kimi_400_invalid_url"
    if "total message size" in lowered and "exceeds limit" in lowered:
        return "kimi_400_message_too_large"
    if "request exceeded model token limit" in lowered:
        return "kimi_400_token_limit"
    if "reasoning_content" in lowered and "missing" in lowered:
        return "kimi_400_missing_reasoning_content"
    if "unsupported image url" in lowered:
        return "kimi_400_unsupported_image_url"
    if "function name" in lowered and "duplicated" in lowered:
        return "kimi_400_duplicate_function_name"
    if "request was rejected" in lowered and "high risk" in lowered:
        return "kimi_400_high_risk_rejected"
    return "kimi_400_unknown"


def _current_turn_has_tool_result(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    last_user = -1
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            last_user = index
    return any(
        isinstance(message, Mapping) and message.get("role") == "tool"
        for message in messages[last_user + 1 :]
    )


def enforce_initial_tool_choice(payload: object) -> tuple[object, bool]:
    """Force one tool call at the start of each user turn.

    The returned bool is metadata only and is safe to log; the request body is not.
    A deep JSON round-trip is unnecessary because only the top-level field changes.
    """

    if not isinstance(payload, dict):
        raise ValueError("chat completion payload must be a JSON object")
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False
    if _current_turn_has_tool_result(payload.get("messages")):
        return payload, False
    transformed = dict(payload)
    transformed["tool_choice"] = "required"
    return transformed, True


def _copy_request_headers(headers: Mapping[str, str], body_size: int) -> dict[str, str]:
    copied = {
        name: value
        for name, value in headers.items()
        if name.casefold() not in _HOP_BY_HOP
        and name.casefold() not in {"host", "content-length"}
    }
    copied["Content-Length"] = str(body_size)
    copied["Accept-Encoding"] = "identity"
    return copied


def _copy_response_headers(response: HTTPResponse) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in response.getheaders()
        if name.casefold() not in _HOP_BY_HOP
        and name.casefold() not in {"content-length"}
    ]


@dataclass(frozen=True)
class ProxyStats:
    requests: int
    forced_requests: int
    error_codes: tuple[str, ...]


class _ProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        upstream_url: str,
        allow_insecure_test_upstream: bool,
    ) -> None:
        super().__init__(server_address, _ProxyHandler)
        parsed = urlparse(upstream_url)
        if not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("invalid upstream URL")
        if not allow_insecure_test_upstream and upstream_url != OFFICIAL_CHAT_COMPLETIONS_URL:
            raise ValueError("live proxy upstream must be the official Kimi endpoint")
        if parsed.scheme not in ({"http", "https"} if allow_insecure_test_upstream else {"https"}):
            raise ValueError("invalid upstream scheme")
        self.upstream_url = upstream_url
        self.upstream = parsed
        self.requests_seen = 0
        self.forced_requests = 0
        self.error_codes: list[str] = []
        self.stats_lock = threading.Lock()

    def record(self, forced: bool) -> None:
        with self.stats_lock:
            self.requests_seen += 1
            self.forced_requests += int(forced)

    def record_error(self, code: str) -> None:
        with self.stats_lock:
            if code not in self.error_codes:
                self.error_codes.append(code)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _ProxyHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        # BaseHTTPRequestHandler logs request details to stderr. This proxy is
        # deliberately silent so credentials and request content cannot leak.
        return

    def _reject(self, status: int) -> None:
        body = b'{"error":"proxy_request_rejected"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802
        expected_path = self.server.upstream.path
        if self.path != expected_path:
            self._reject(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._reject(400)
            return
        if content_length < 1 or content_length > _MAX_REQUEST_BYTES:
            self._reject(413)
            return
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body)
            transformed, forced = enforce_initial_tool_choice(payload)
            outgoing = json.dumps(
                transformed, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            self._reject(400)
            return

        self.server.record(forced)
        upstream = self.server.upstream
        connection_type = HTTPSConnection if upstream.scheme == "https" else HTTPConnection
        connection = connection_type(
            upstream.hostname,
            upstream.port,
            timeout=900,
            context=ssl.create_default_context() if upstream.scheme == "https" else None,
        ) if upstream.scheme == "https" else connection_type(
            upstream.hostname, upstream.port, timeout=900
        )
        try:
            connection.request(
                "POST",
                upstream.path,
                body=outgoing,
                headers=_copy_request_headers(self.headers, len(outgoing)),
            )
            response = connection.getresponse()
            error_body = bytearray()
            self.send_response(response.status, response.reason)
            for name, value in _copy_response_headers(response):
                self.send_header(name, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                if response.status == 400 and len(error_body) < _MAX_ERROR_CLASSIFICATION_BYTES:
                    remaining = _MAX_ERROR_CLASSIFICATION_BYTES - len(error_body)
                    error_body.extend(chunk[:remaining])
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            if response.status == 400:
                self.server.record_error(_classify_kimi_400(bytes(error_body)))
        except (OSError, ConnectionError):
            if not self.wfile.closed:
                self.close_connection = True
        finally:
            connection.close()


class InitialToolChoiceProxy(AbstractContextManager["InitialToolChoiceProxy"]):
    """Loopback-only, body-silent proxy for the official Kimi coding endpoint."""

    def __init__(
        self,
        *,
        upstream_url: str = OFFICIAL_CHAT_COMPLETIONS_URL,
        _allow_insecure_test_upstream: bool = False,
    ) -> None:
        self._server = _ProxyHTTPServer(
            ("127.0.0.1", 0),
            upstream_url=upstream_url,
            allow_insecure_test_upstream=_allow_insecure_test_upstream,
        )
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        port = self._server.server_address[1]
        path = self._server.upstream.path.removesuffix("/chat/completions")
        return f"http://127.0.0.1:{port}{path}"

    @property
    def stats(self) -> ProxyStats:
        with self._server.stats_lock:
            return ProxyStats(
                requests=self._server.requests_seen,
                forced_requests=self._server.forced_requests,
                error_codes=tuple(self._server.error_codes),
            )

    def __enter__(self) -> "InitialToolChoiceProxy":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="anchor-initial-tool-proxy",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

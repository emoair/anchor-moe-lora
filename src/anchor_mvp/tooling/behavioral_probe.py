"""Offline loopback behavioral probe for a patched OpenCode executable."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from typing import Any, Mapping

from .config import AGENT_ID


PROBE_MARKER = "anchor-offline-probe-marker-v1"
DEFAULT_PROBE_TIMEOUT_SECONDS = 300.0


@dataclass
class ProbeTranscript:
    requests: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def validate(self) -> tuple[bool, str]:
        if self.error:
            return False, self.error
        if len(self.requests) != 2:
            return False, f"behavioral probe expected 2 requests, observed {len(self.requests)}"
        first, second = self.requests
        tools = first.get("tools")
        names = {
            str(item.get("function", {}).get("name", ""))
            for item in tools
            if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
        } if isinstance(tools, list) else set()
        if first.get("tool_choice") not in {None, "auto"} or "read" not in names:
            return False, "first provider request did not preserve automatic tool choice"
        if names != {"read"}:
            return False, "first provider request exposed tools outside the local read allowlist"
        if second.get("tool_choice") not in {None, "auto"}:
            return False, "second provider request did not preserve automatic tool choice"
        messages = second.get("messages")
        if not isinstance(messages, list):
            return False, "second provider request has no messages"
        tool_call_messages = [
            item
            for item in messages
            if isinstance(item, Mapping)
            and item.get("role") == "assistant"
            and isinstance(item.get("tool_calls"), list)
        ]
        if not tool_call_messages or any(
            "reasoning_content" not in item or not isinstance(item.get("reasoning_content"), str)
            for item in tool_call_messages
        ):
            return False, "second provider request omitted reasoning_content from assistant tool call"
        results = [item for item in messages if isinstance(item, Mapping) and item.get("role") == "tool"]
        if not results or not any(PROBE_MARKER in str(item.get("content", "")) for item in results):
            return False, "second provider request did not contain the completed probe tool result"
        return True, "automatic tool choice, reasoning replay, and tool result behavior verified"


def _response(transcript: ProbeTranscript, request: Mapping[str, Any]) -> dict[str, Any]:
    index = len(transcript.requests)
    if index == 1:
        tools = request.get("tools")
        read_name = "read"
        if isinstance(tools, list):
            available = [
                str(item.get("function", {}).get("name", ""))
                for item in tools
                if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
            ]
            if "read" not in available:
                transcript.error = "probe request did not expose read"
            elif read_name not in available:
                read_name = available[0]
        return {
            "id": "chatcmpl-anchor-probe-1",
            "object": "chat.completion",
            "created": 0,
            "model": "probe-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_anchor_probe_1",
                                "type": "function",
                                "function": {
                                    "name": read_name,
                                    "arguments": json.dumps({"filePath": "probe.txt"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    return {
        "id": "chatcmpl-anchor-probe-2",
        "object": "chat.completion",
        "created": 0,
        "model": "probe-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "probe complete"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _title_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-anchor-probe-title",
        "object": "chat.completion",
        "created": 0,
        "model": "probe-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Anchor probe"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _is_title_request(request: Mapping[str, Any]) -> bool:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, Mapping)
        and message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and message["content"].startswith("You are a title generator.")
        for message in messages
    )


def _is_execution_request(request: Mapping[str, Any]) -> bool:
    """Identify actual model turns by the local tool schema, not arrival order."""

    return isinstance(request.get("tools"), list)


def _stream_chunk(response: Mapping[str, Any], *, delta: Mapping[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"delta": dict(delta or {})}
    if finish:
        choice["finish_reason"] = finish
    return {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
        "choices": [choice],
    }


def _sse_lines(response: Mapping[str, Any]) -> list[str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("probe response has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("probe response has no assistant message")
    lines = [f"data: {json.dumps(_stream_chunk(response, delta={'role': 'assistant'}))}\n\n"]
    content = message.get("content")
    if isinstance(content, str) and content:
        lines.append(f"data: {json.dumps(_stream_chunk(response, delta={'content': content}))}\n\n")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls):
            if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                raise ValueError("probe tool call is malformed")
            function = call["function"]
            name = function.get("name")
            if not isinstance(name, str):
                raise ValueError("probe tool call has no name")
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                raise ValueError("probe tool call arguments are malformed")
            lines.append(
                f"data: {json.dumps(_stream_chunk(response, delta={'tool_calls': [{'index': index, 'id': call.get('id', f'call_{index}'), 'type': 'function', 'function': {'name': name, 'arguments': ''}}]}))}\n\n"
            )
            if arguments:
                lines.append(
                    f"data: {json.dumps(_stream_chunk(response, delta={'tool_calls': [{'index': index, 'function': {'arguments': arguments}}]}))}\n\n"
                )
    finish = choice.get("finish_reason")
    if not isinstance(finish, str):
        raise ValueError("probe response has no finish reason")
    lines.append(f"data: {json.dumps(_stream_chunk(response, finish=finish))}\n\n")
    lines.append("data: [DONE]\n\n")
    return lines


def run_behavioral_probe(
    executable: Path,
    *,
    probe_root: Path,
    environment: Mapping[str, str],
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    probe_root.mkdir(parents=True, exist_ok=True)
    workspace = probe_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "probe.txt").write_text(PROBE_MARKER + "\n", encoding="utf-8")
    transcript = ProbeTranscript()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"object": "list", "data": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("request is not an object")
                if _is_title_request(value):
                    response = _title_response()
                elif _is_execution_request(value):
                    transcript.requests.append(value)
                    response = _response(transcript, value)
                else:
                    raise ValueError("probe received a non-title request without local tools")
                if value.get("stream") is True:
                    body = "".join(_sse_lines(response)).encode("utf-8")
                    content_type = "text/event-stream"
                else:
                    body = json.dumps(response).encode("utf-8")
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                transcript.error = f"loopback provider error: {type(exc).__name__}"
                self.send_error(400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        config = {
            "model": "anchor-probe/probe-model",
            "default_agent": AGENT_ID,
            "share": "disabled",
            "provider": {
                "anchor-probe": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "offline-probe"},
                    "models": {
                        "probe-model": {
                            "name": "Offline probe",
                            "reasoning": True,
                            "interleaved": {"field": "reasoning_content"},
                        }
                    },
                }
            },
            "permission": {"*": "deny", "read": "allow", "external_directory": "deny"},
            "agent": {
                AGENT_ID: {
                    "mode": "primary",
                    "steps": 3,
                    "permission": {"*": "deny", "read": "allow"},
                }
            },
        }
        config_path = probe_root / "opencode.probe.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        child_env = dict(environment)
        child_env.pop("OPENCODE_CONFIG_CONTENT", None)
        child_env.update(
            {
                "OPENCODE_CONFIG": str(config_path),
                "OPENCODE_CONFIG_DIR": str(probe_root / "config"),
                "XDG_CONFIG_HOME": str(probe_root / "config"),
                "XDG_DATA_HOME": str(probe_root / "data"),
                "XDG_CACHE_HOME": str(probe_root / "cache"),
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
                "OPENCODE_AUTO_SHARE": "false",
            }
        )
        completed = subprocess.run(
            [
                str(executable),
                "run",
                "--format",
                "json",
                "--model",
                "anchor-probe/probe-model",
                "--agent",
                AGENT_ID,
                "--dir",
                str(workspace),
                "Read probe.txt once, then finish.",
            ],
            cwd=workspace,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            return False, f"behavioral probe process exited {completed.returncode}"
        return transcript.validate()
    except subprocess.TimeoutExpired:
        return False, "behavioral probe timed out"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

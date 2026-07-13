"""Offline Responses wire probe for audited custom OpenCode providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from typing import Any, Mapping

from .config import AGENT_ID, OpenCodeProvider


WIRE_MARKER = "anchor-responses-wire-probe-v1"
WIRE_CALL_ID = "call_anchor_read"
# These fields and only these fields are present in the independently verified
# Ark/OpenCode Responses request contract.  The two SDK-added fields below are
# also value-locked in ``ResponsesWireTranscript.validate``; accepting them by
# name alone would turn an upstream SDK change into an unaudited live request.
ARK_VERIFIED_RESPONSE_FIELDS = frozenset(
    {
        "include",
        "input",
        "max_output_tokens",
        "model",
        "reasoning",
        "store",
        "stream",
        "tool_choice",
        "tools",
    }
)
ARK_VERIFIED_INCLUDE = ("reasoning.encrypted_content",)
ARK_VERIFIED_TOOLS = ({"type": "function", "name": "read", "parameters": {}},)


def _walk(value: object):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


@dataclass
class ResponsesWireTranscript:
    paths: list[str] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def validate(self, provider: OpenCodeProvider) -> tuple[bool, str]:
        if self.error:
            return False, self.error
        if len(self.requests) != 2:
            return False, (
                "Responses wire probe expected two tool turns, observed "
                f"{len(self.requests)}"
            )
        if len(self.paths) != 2:
            return False, (
                "Responses wire probe expected two request paths, observed "
                f"{len(self.paths)}"
            )
        if any(not path.endswith("/responses") for path in self.paths):
            return False, "Responses provider did not use the /responses endpoint"
        first, second = self.requests
        unknown = sorted(
            set().union(*(set(request) for request in self.requests))
            - ARK_VERIFIED_RESPONSE_FIELDS
        )
        if unknown:
            return False, "unverified Responses wire fields: " + ",".join(unknown)
        for turn, request in enumerate(self.requests, start=1):
            if request.get("model") != provider.model:
                return (
                    False,
                    f"Responses wire model drift on turn {turn}: expected audited model",
                )
            if request.get("reasoning") != {"effort": "max"}:
                return (
                    False,
                    f"Responses wire reasoning drift on turn {turn}: expected max",
                )
            tools = request.get("tools")
            if not isinstance(tools, list) or tuple(tools) != ARK_VERIFIED_TOOLS:
                return (
                    False,
                    "Responses wire tools drift on turn "
                    f"{turn}: expected the exact local read allowlist",
                )
            if request.get("tool_choice") != "auto":
                return (
                    False,
                    f"Responses wire tool_choice drift on turn {turn}: expected auto",
                )
            if request.get("store") is not False:
                return (
                    False,
                    f"Responses wire store drift on turn {turn}: expected false",
                )
            include = request.get("include")
            if not isinstance(include, list) or tuple(include) != ARK_VERIFIED_INCLUDE:
                return (
                    False,
                    "Responses wire include drift on turn "
                    f'{turn}: expected ["reasoning.encrypted_content"]',
                )
        tool_outputs = [
            item
            for item in _walk(second.get("input"))
            if item.get("type") == "function_call_output"
        ]
        if len(tool_outputs) != 1:
            return (
                False,
                "Responses second turn must contain exactly one local tool result",
            )
        tool_output = tool_outputs[0]
        if tool_output.get("call_id") != WIRE_CALL_ID:
            return (
                False,
                "Responses second turn tool result call_id did not match the first call",
            )
        if WIRE_MARKER not in str(tool_output.get("output", "")):
            return (
                False,
                "Responses second turn omitted the completed local tool result marker",
            )
        return True, "Responses endpoint, max effort, and tool-result replay verified"


def _event_lines(events: list[dict[str, object]]) -> bytes:
    return "".join(
        f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
    ).encode("utf-8")


def _created(model: str) -> dict[str, object]:
    return {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_anchor_wire",
            "created_at": 0,
            "model": model,
            "service_tier": None,
        },
    }


def _completed(sequence: int) -> dict[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": sequence,
        "response": {
            "incomplete_details": None,
            "service_tier": None,
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    }


def _tool_events(model: str) -> list[dict[str, object]]:
    arguments = json.dumps({"filePath": "probe.txt"}, separators=(",", ":"))
    item = {
        "type": "function_call",
        "id": "item_anchor_read",
        "call_id": WIRE_CALL_ID,
        "name": "read",
        "arguments": arguments,
        "status": "completed",
    }
    return [
        _created(model),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "item_anchor_read",
            "delta": arguments,
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": "item_anchor_read",
            "arguments": arguments,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": item,
        },
        _completed(5),
    ]


def _text_events(model: str, text: str) -> list[dict[str, object]]:
    return [
        _created(model),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "item_anchor_text",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "item_anchor_text",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "item_anchor_text",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
        },
        _completed(4),
    ]


def run_responses_wire_probe(
    executable: Path,
    *,
    probe_root: Path,
    environment: Mapping[str, str],
    provider: OpenCodeProvider,
    timeout_seconds: float = 45.0,
) -> tuple[bool, str]:
    """Capture an offline two-turn Responses request and fail on extra fields."""

    if not provider.is_responses:
        return True, "Responses wire probe not required for this provider"
    probe_root = probe_root.resolve()
    probe_root.mkdir(parents=True, exist_ok=True)
    workspace = probe_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "probe.txt").write_text(WIRE_MARKER + "\n", encoding="utf-8")
    transcript = ResponsesWireTranscript()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            body = b'{"object":"list","data":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request is not an object")
                tools = request.get("tools")
                if isinstance(tools, list) and tools:
                    transcript.paths.append(self.path)
                    transcript.requests.append(request)
                    events = (
                        _tool_events(provider.model)
                        if len(transcript.requests) == 1
                        else _text_events(provider.model, "probe complete")
                    )
                else:
                    events = _text_events(provider.model, "Anchor probe")
                body = _event_lines(events)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                transcript.error = f"Responses loopback error: {type(error).__name__}"
                self.send_error(400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        config = {
            "model": f"{provider.provider_id}/{provider.model}",
            "default_agent": AGENT_ID,
            "share": "disabled",
            "provider": {
                provider.provider_id: {
                    "npm": provider.npm,
                    "options": {
                        "baseURL": f"http://127.0.0.1:{port}/v1",
                        "apiKey": "offline-wire-probe",
                        "setCacheKey": False,
                    },
                    "models": {
                        provider.model: {
                            "name": "Offline Responses probe",
                            "reasoning": True,
                            "limit": {"context": 128000, "output": 32768},
                            "variants": {
                                provider.variant: {
                                    "reasoningEffort": provider.variant,
                                }
                            },
                        }
                    },
                }
            },
            "permission": {"*": "deny", "read": "allow"},
            "agent": {
                AGENT_ID: {
                    "mode": "primary",
                    "steps": 3,
                    "permission": {"*": "deny", "read": "allow"},
                }
            },
        }
        config_path = probe_root / "opencode.responses-probe.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        child_env = dict(environment)
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
                f"{provider.provider_id}/{provider.model}",
                "--agent",
                AGENT_ID,
                "--variant",
                provider.variant,
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
            return False, f"Responses wire probe process exited {completed.returncode}"
        return transcript.validate(provider)
    except subprocess.TimeoutExpired:
        return False, "Responses wire probe timed out"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

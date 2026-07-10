from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from .models import ToolTraceEntry
from .policy import ToolPolicy


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def classify_error_text(value: str) -> tuple[str, ...]:
    lowered = value.lower()
    codes: list[str] = []
    if "invalid_url" in lowered or "invalid url" in lowered or "missing scheme" in lowered:
        codes.append("invalid_url")
    if "context canceled" in lowered or "context cancelled" in lowered or "http 499" in lowered:
        codes.append("client_cancelled")
    if "rate limit" in lowered or "status code: 429" in lowered or "http 429" in lowered:
        codes.append("rate_limited")
    if "invalid authentication" in lowered or "status code: 401" in lowered or "http 401" in lowered:
        codes.append("authentication_failed")
    return tuple(codes)


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_string(mapping: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str):
            return value
    return None


def _first_integer(mapping: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_not_none(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def parse_opencode_jsonl(
    stdout: str, policy: ToolPolicy
) -> tuple[tuple[ToolTraceEntry, ...], int]:
    """Reduce raw OpenCode events to safe metadata.

    Model text, thinking blocks, file contents, and arbitrary tool arguments are
    deliberately discarded. Unapproved commands are represented only by a hash.
    """

    trace: list[ToolTraceEntry] = []
    rejected = 0
    seen: set[tuple[str, str | None, int | None, str]] = set()
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in _walk_dicts(event):
            event_type = str(item.get("type", "")).lower()
            tool = _first_string(item, ("tool", "toolName", "tool_name", "name"))
            if not tool or not (
                event_type in {"tool", "tool_use", "tool_call", "tool_result"}
                or "state" in item
                or "input" in item
            ):
                continue
            tool = tool.lower()
            state = item.get("state") if isinstance(item.get("state"), dict) else {}
            tool_input = item.get("input") if isinstance(item.get("input"), dict) else {}
            state_input = state.get("input") if isinstance(state.get("input"), dict) else {}
            command = (
                _first_string(item, ("command", "cmd"))
                or _first_string(tool_input, ("command", "cmd"))
                or _first_string(state_input, ("command", "cmd"))
            )
            exit_code = _first_not_none(
                _first_integer(item, ("exitCode", "exit_code", "code")),
                _first_integer(state, ("exitCode", "exit_code", "code")),
            )
            status = str(state.get("status", item.get("status", "observed")))
            safe_command: str | None = None
            command_hash: str | None = None
            allowed = policy.is_tool_allowed(tool)
            if command is not None:
                normalized = policy.normalize_command(command)
                allowed = allowed and policy.is_command_allowed(command)
                if allowed:
                    safe_command = normalized
                else:
                    command_hash = policy.command_digest(command)
            if not allowed:
                rejected += 1
                status = "rejected"
            key = (tool, safe_command or command_hash, exit_code, status)
            if key in seen:
                continue
            seen.add(key)
            trace.append(
                ToolTraceEntry(
                    sequence=len(trace) + 1,
                    source="agent",
                    tool=tool,
                    status=status,
                    command=safe_command,
                    command_sha256=command_hash,
                    exit_code=exit_code,
                )
            )
    return tuple(trace), rejected

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator

from .models import PublicDecisionStep, PublicOutcome, ToolTraceEntry
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


_SAFE_ERROR_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")


def classify_error_metadata(stdout: str, stderr: str) -> tuple[str, ...]:
    """Classify failures without retaining messages, model text, or tool contents."""

    codes = list(classify_error_text(stdout + "\n" + stderr))
    status_names = {"status", "statuscode", "httpstatus", "http_status"}
    code_names = {"code", "name", "errorcode", "error_code"}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in _walk_dicts(event):
            if str(item.get("type", "")).casefold() == "error":
                codes.append("agent_error_event")
            item_codes = {
                str(value).casefold()
                for key, value in item.items()
                if str(key).casefold() in code_names
                and isinstance(value, str)
                and _SAFE_ERROR_TOKEN.fullmatch(value)
            }
            for key, value in item.items():
                name = str(key).casefold()
                if name in status_names:
                    try:
                        status = int(value)
                    except (TypeError, ValueError):
                        continue
                    if status == 400:
                        codes.append(
                            "invalid_url" if "invalid_url" in item_codes else "invalid_request"
                        )
                    elif status == 401:
                        codes.append("authentication_failed")
                    elif status == 403:
                        codes.append("forbidden")
                    elif status == 404:
                        codes.append("endpoint_or_model_not_found")
                    elif status in {408, 499}:
                        codes.append("client_cancelled")
                    elif status == 429:
                        codes.append("rate_limited")
                    elif 500 <= status <= 599:
                        codes.append("upstream_server_error")
                elif (
                    name in code_names
                    and isinstance(value, str)
                    and _SAFE_ERROR_TOKEN.fullmatch(value)
                ):
                    normalized = value.casefold()
                    if normalized == "invalid_url":
                        codes.append("invalid_url")
                    else:
                        codes.append(f"agent_{normalized}")
    if stderr.strip():
        codes.append("agent_stderr_present")
    return tuple(dict.fromkeys(codes))


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _json_candidate(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def parse_public_outcome(stdout: str) -> PublicOutcome | None:
    """Extract only the explicit public work summary from OpenCode events.

    Arbitrary assistant text and reasoning blocks are ignored. The accepted object
    has a narrow schema and bounded public evidence fields.
    """

    accepted: PublicOutcome | None = None
    forbidden = {"thinking", "reasoning", "chain_of_thought", "cot"}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or str(event.get("type", "")).casefold() != "text":
            continue
        part = event.get("part")
        if not isinstance(part, dict) or str(part.get("type", "")).casefold() not in {
            "",
            "text",
        }:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        for value in (text,):
            candidate = _json_candidate(value)
            if not candidate or candidate.get("schema_version") != "anchor.public-outcome.v1":
                continue
            if forbidden.intersection(str(key).casefold() for key in candidate):
                continue
            status = candidate.get("status")
            trace = candidate.get("decision_trace")
            repairs = candidate.get("repair_summaries")
            summary = candidate.get("final_summary")
            if status not in {"completed", "blocked", "partial"}:
                continue
            if not isinstance(trace, list) or not 1 <= len(trace) <= 8:
                continue
            steps: list[PublicDecisionStep] = []
            valid = True
            for item in trace:
                if not isinstance(item, dict):
                    valid = False
                    break
                fields = tuple(
                    str(item.get(name, "")).strip()
                    for name in ("check", "evidence", "action")
                )
                if not all(fields) or any(len(field) > 600 for field in fields):
                    valid = False
                    break
                steps.append(PublicDecisionStep(*fields))
            if not valid or not isinstance(repairs, list) or len(repairs) > 8:
                continue
            clean_repairs = tuple(str(item).strip() for item in repairs)
            if any(not item or len(item) > 600 for item in clean_repairs):
                continue
            if not isinstance(summary, str) or not summary.strip() or len(summary.strip()) > 1000:
                continue
            accepted = PublicOutcome(
                status=status,
                decision_trace=tuple(steps),
                repair_summaries=clean_repairs,
                final_summary=summary.strip(),
            )
    return accepted
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
